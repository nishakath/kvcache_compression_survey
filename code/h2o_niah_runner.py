import torch
import argparse
import numpy as np
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForCausalLM
from h2o_utils import generate_with_h2o

def generate_random_needle():
    """Generate a random needle with a 6-digit magic number."""
    magic = random.randint(100000, 999999)
    return f"The magic number is {magic}.", str(magic)

def plot_survival_map(survival_history, total_len, budget, label="eviction"):
    """
    Visualize which token indices survived across generation steps.
    """
    plt.figure(figsize=(10, 6))
    
    # Create a survival matrix: (steps, total_len)
    steps = len(survival_history)
    matrix = np.zeros((steps, total_len))
    
    for i, kept_indices in enumerate(survival_history):
        for idx in kept_indices:
            if idx < total_len:
                matrix[i, idx] = 1
                
    plt.imshow(matrix, aspect='auto', interpolation='nearest', cmap='Blues', origin='lower')
    plt.colorbar(label='Kept in Cache')
    plt.xlabel('Token Index')
    plt.ylabel('Eviction Step')
    plt.title(f'H2O Token Survival Map (Budget: {budget})')
    
    filename = f"h2o_survival_{label}.png"
    plt.savefig(filename)
    print(f"Survival map saved as: {filename}")

def calculate_effective_context(results, lengths):
    """
    Determine the length at which accuracy drops below 80%.
    """
    # Group by length and calculate average success
    success_by_len = {l: [] for l in lengths}
    for res in results:
        success_by_len[res['len']].append(1 if res['success'] else 0)
    
    avg_success = {l: np.mean(v) for l, v in success_by_len.items()}
    
    effective_len = lengths[0]
    for l in sorted(lengths):
        if avg_success[l] >= 0.8:
            effective_len = l
        else:
            break
            
    return effective_len, avg_success

def build_prompt(tokenizer, haystack_text, needle_text, question):
    """
    Build a properly formatted prompt. Uses chat template if available.
    """
    full_text = haystack_text + "\n" + needle_text + "\n" + haystack_text
    
    # Try to use the model's chat template (works for Llama-2-chat, etc.)
    try:
        messages = [
            {"role": "user", "content": f"Read the following text carefully:\n\n{full_text}\n\n{question}"}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        # Fallback: raw prompt
        prompt = f"{full_text}\n\n{question}\nAnswer:"
    
    return prompt


def run_niah_experiment(model_id, budget, lengths, depths, device="cuda", baseline=False):
    print(f"Loading model: {model_id} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    if device == "cpu":
        dtype = torch.float32
    else:
        dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=dtype, 
        device_map=None if device == "cpu" else "auto"
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()
    
    mode_str = "BASELINE (no eviction)" if baseline else f"H2O Budget: {budget}"
    results = []
    print(f"\nStarting NIAH Benchmark ({mode_str})...")
    print("-" * 90)
    
    first_survival_captured = False

    for length in lengths:
        for depth in depths:
            needle_text, expected_answer = generate_random_needle()
            question = "What is the magic number mentioned in the text? Reply with ONLY the number, nothing else."
            
            # Build haystack
            filler = "The weather in San Francisco is foggy today. "
            filler_tokens = tokenizer.encode(filler, add_special_tokens=False)
            num_repeats = max(1, length // len(filler_tokens))
            haystack_tokens = (filler_tokens * num_repeats)[:length]
            haystack_before = tokenizer.decode(haystack_tokens[:int(len(haystack_tokens) * depth)])
            haystack_after = tokenizer.decode(haystack_tokens[int(len(haystack_tokens) * depth):])
            
            # Build prompt with chat template
            full_text = haystack_before + " " + needle_text + " " + haystack_after
            try:
                messages = [
                    {"role": "user", "content": f"Read the following text carefully and answer the question.\n\nText: {full_text}\n\nQuestion: {question}"}
                ]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                prompt = f"{full_text}\n\n{question}\nAnswer:"
            
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
            actual_len = input_ids.shape[1]
            
            if baseline:
                use_budget = actual_len + 5000 # Large enough
            else:
                use_budget = budget
            
            all_tokens, stats = generate_with_h2o(
                model, input_ids,
                max_new_tokens=20,
                budget=use_budget,
            )
            
            # Visualize the very first case that actually triggers eviction
            if not baseline and not first_survival_captured and actual_len > budget:
                plot_survival_map(stats['survival_history'], actual_len, budget, label=f"len{length}")
                first_survival_captured = True

            response = tokenizer.decode(all_tokens[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
            
            success = expected_answer in response
            results.append({"len": length, "depth": depth, "success": success, "resp": response})
            
            status = "PASS" if success else "FAIL"
            print(f"[{status}] Len:{length:6} (actual:{actual_len:5}) | Depth:{depth:.1f} | "
                  f"Needle:{expected_answer} | Resp: {response[:30]:30s} | "
                  f"Evictions:{stats['evictions']} Cache:{stats['cache_len']}")
            
    return results, lengths, depths


def plot_results(results, lengths, depths, budget, baseline=False):
    accuracy_matrix = np.zeros((len(depths), len(lengths)))
    for res in results:
        i = depths.index(res['depth'])
        j = lengths.index(res['len'])
        accuracy_matrix[i, j] = 1 if res['success'] else 0
    
    title_suffix = "Baseline (No Eviction)" if baseline else f"H2O Budget: {budget}"
    
    plt.figure(figsize=(12, 7))
    sns.heatmap(accuracy_matrix, annot=True, fmt=".0f",
                xticklabels=lengths, yticklabels=[f"{d:.1f}" for d in depths], 
                cmap="RdYlGn", vmin=0, vmax=1, cbar_kws={"label": "Retrieval Success"})
    
    # Calculate effective context length
    eff_len, avg_acc = calculate_effective_context(results, lengths)
    
    plt.title(f"NIAH Retrieval — {title_suffix}\nEffective Context Length: ~{eff_len} tokens")
    plt.xlabel("Context Length (Tokens)")
    plt.ylabel("Needle Depth")
    
    tag = "baseline" if baseline else str(budget)
    filename = f"h2o_niah_results_{tag}.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"\nHeatmap saved as: {filename}")
    
    print(f"\nSummary of Retrieval Accuracy:")
    for l in sorted(lengths):
        print(f"  Length {l:5}: {avg_acc[l]*100:3.0f}%")
    print(f"Estimated Effective Context Window: {eff_len} tokens (where accuracy >= 80%)\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="H2O NIAH Experiment")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--budget", type=int, default=1024)
    parser.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    parser.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.3, 0.5, 0.7, 0.9])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--baseline", action="store_true", help="Run without H2O eviction (baseline)")
    
    args = parser.parse_args()
    
    results, lengths, depths = run_niah_experiment(
        args.model, args.budget, args.lengths, args.depths, 
        device=args.device, baseline=args.baseline
    )
    plot_results(results, lengths, depths, args.budget, baseline=args.baseline)
    print("\nDone!")

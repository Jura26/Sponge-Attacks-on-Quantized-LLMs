import argparse
import os
import time

import random
import string
from monitoring import SystemMonitor
try:
    from model import load_model_and_tokenizer, cleanup_model
    import torch
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Hint: Check if 'backend.model' or 'torch' imports are failing.")
    # sys.exit(1) # Don't exit immediately so we can see more info if running via uvicorn

# --- Configuration ---
POPULATION_SIZE = 10
GENERATIONS = 5
PROMPT_LENGTH = 20  # Starting length (short, to amplify output/load)
MUTATION_RATE = 0.2
MODEL_ID = "gpt2"   # Default, can be overridden
# Define a range for dynamic token generation
MAX_NEW_TOKENS = 0  # 0 disables the cap (use the full context window)



def _effective_context_limit(model) -> int:
    context_limit = getattr(model.config, "max_position_embeddings", None)
    if context_limit is None:
        context_limit = getattr(model.config, "n_positions", None)

    max_override = int(os.environ.get("SPONGE_MAX_CONTEXT", 0))
    if max_override > 0:
        return min(context_limit, max_override) if context_limit else max_override

    gguf_ctx = int(os.environ.get("SPONGE_GGUF_CTX", 0))
    if gguf_ctx > 0:
        return min(context_limit, gguf_ctx) if context_limit else gguf_ctx

    return context_limit or 4096

def warmup(model, tokenizer, device, target_temp=85, max_wait=300, model_max_length=None):
    """Run inference in a loop until GPU reaches target temperature."""
    if model_max_length is None:
        model_max_length = 4096

    temp_monitor = SystemMonitor(device=device)
    current_temp = temp_monitor._get_temp()

    if current_temp == 0:
        print("  🔥 Warmup skipped (temp sensor unavailable)", flush=True)
        return
    if current_temp >= target_temp:
        print(f"  🔥 Already at {current_temp}°C, skipping warmup", flush=True)
        return

    print(f"  🔥 Warming up GPU to {target_temp}°C (currently {current_temp}°C)...", flush=True)

    # Same limits as main eval loop — avoids shape mismatches on GGUF backend
    max_input_len = model_max_length - 50

    start_time = time.time()
    iteration = 0

    while time.time() - start_time < max_wait:
        current_temp = temp_monitor._get_temp()
        print(f"     [{iteration}] {current_temp}°C", end="\r", flush=True)

        if current_temp >= target_temp:
            print(f"  🔥 Warmup done — {current_temp}°C after {iteration} runs     ", flush=True)
            return

        try:
            # ✅ Use same prompt generation + tokenization as main eval — guaranteed compatible
            warmup_prompt = generate_random_prompt(tokenizer, length=PROMPT_LENGTH)

            inputs = tokenizer(
                warmup_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_len
            ).to(device)

            input_len = inputs.input_ids.shape[1]
            safe_max_new_tokens = max(1, model_max_length - input_len - 1)

            with torch.no_grad():
                model.generate(
                    **inputs,
                    max_new_tokens=min(128, safe_max_new_tokens),  # short bursts = faster heating
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

        except Exception as e:
            print(f"\n  ⚠️ Warmup error: {e}", flush=True)
            break

        iteration += 1

    current_temp = temp_monitor._get_temp()
    print(f"  ⚠️ Warmup timed out at {current_temp}°C after {iteration} runs", flush=True)

def cooldown(target_temp=60, max_wait=10, device="cpu"):
    """Wait for hardware to cool down to ensure fair testing."""
    print(f"  Cooling down (target < {target_temp}C)...", end="", flush=True)
    temp_monitor = SystemMonitor(device=device)
    
    # Check if sensors are working at all
    initial_temp = temp_monitor._get_temp()
    if initial_temp == 0:
        print(" Skipped (temp sensor unavailable)")
        return

    for _ in range(max_wait):
        current_temp = temp_monitor._get_temp()
        if current_temp == 0:
            print(" Skipped (temp sensor lost)")
            return
        if current_temp < target_temp:
            print(f" Done ({current_temp}C)")
            return
        time.sleep(1)
        print(".", end="", flush=True)
    print(f" Timeout ({current_temp}C)")

# --- Genetic Algorithm ---

def generate_random_prompt(tokenizer=None, length=20):
    if tokenizer:
        # Generate random tokens from vocabulary
        vocab_size = tokenizer.vocab_size
        # Avoid special tokens (usually at start/end of vocab, but varies by model)
        # Simple heuristic: range(100, vocab_size-100)
        random_ids = [random.randint(100, vocab_size - 100) for _ in range(length)]
        return tokenizer.decode(random_ids, skip_special_tokens=True)
    
    chars = string.ascii_letters + string.digits + " "
    return "".join(random.choice(chars) for _ in range(length))

def mutate(prompt, tokenizer=None):
    """Randomly change, add, or remove tokens (preferred) or characters."""
    if tokenizer:
        try:
            tokens = tokenizer.encode(prompt, add_special_tokens=False)
            if not tokens: tokens = [random.randint(100, tokenizer.vocab_size-100)]
            
            vocab_size = tokenizer.vocab_size
            
            if random.random() < 0.5 and len(tokens) > 1:
                # Swap/Change Token
                idx = random.randint(0, len(tokens)-1)
                tokens[idx] = random.randint(100, vocab_size-100)
            elif random.random() < 0.5:
                # Add Token
                idx = random.randint(0, len(tokens))
                tokens.insert(idx, random.randint(100, vocab_size-100))
            else:
                # Remove Token
                if len(tokens) > 2:
                    idx = random.randint(0, len(tokens)-1)
                    while idx < len(tokens):
                        tokens.pop(idx)
                        break
            
            return tokenizer.decode(tokens, skip_special_tokens=True)
        except:
            pass # Fallback to char mutation if encoding fails
            
    chars = string.ascii_letters + string.digits + " "
    prompt_list = list(prompt)
    
    if random.random() < 0.5 and len(prompt_list) > 1:
        # Swap/Change
        idx = random.randint(0, len(prompt_list)-1)
        prompt_list[idx] = random.choice(chars)
    elif random.random() < 0.5:
        # Add char
        idx = random.randint(0, len(prompt_list))
        prompt_list.insert(idx, random.choice(chars))
    else:
        # Remove char (if long enough)
        if len(prompt_list) > 5:
            idx = random.randint(0, len(prompt_list)-1)
            prompt_list.pop(idx)
            
    return "".join(prompt_list)

def crossover(p1, p2, tokenizer=None):
    """Combine two prompts. Prefer token-boundary split if tokenizer provided."""
    if tokenizer:
        try:
            t1 = tokenizer.encode(p1, add_special_tokens=False)
            t2 = tokenizer.encode(p2, add_special_tokens=False)
            if len(t1) > 1 and len(t2) > 1:
                split = random.randint(1, min(len(t1), len(t2)) - 1)
                new_tokens = t1[:split] + t2[split:]
                return tokenizer.decode(new_tokens, skip_special_tokens=True)
        except:
            pass
            
    # String fallback
    split = random.randint(1, min(len(p1), len(p2)) - 1)
    return p1[:split] + p2[split:]

def evaluate_population(population, model, tokenizer, device, progress_callback=None):
    scores = []

    print(f"\nEvaluating {len(population)} prompts...", flush=True)
    if progress_callback:
        progress_callback({"status": "eval", "message": f"Evaluating {len(population)} prompts..."})

    model_max_length = _effective_context_limit(model)
    warmup(model, tokenizer, device, target_temp=85, max_wait=300, model_max_length=model_max_length)

    for i, prompt in enumerate(population):
        cooldown(target_temp=85, max_wait=5, device=device)
        if progress_callback:
            progress_callback({"status": "eval", "message": f"  Cooling down before prompt {i+1}/{len(population)}..."})

        print(f"  [{i+1}/{len(population)}] Testing: '{prompt[:30]}...'", flush=True)
        if progress_callback:
            progress_callback({"status": "eval", "message": f"  [{i+1}/{len(population)}] Testing: '{prompt[:30]}...'"})

        # ── These are INSIDE the for-i loop ──────────────────────────────
        runs = 3
        run_results = []

        for r in range(runs):
            generated_tokens = 0
            generated_text = ""
            input_len = 0

            monitor = SystemMonitor(device=device)
            monitor.start()

            try:
                max_input_len = model_max_length - 50
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_input_len
                ).to(device)

                input_len = inputs.input_ids.shape[1]

                if "attention_mask" not in inputs:
                    inputs["attention_mask"] = torch.ones_like(inputs.input_ids)

                remaining_context = model_max_length - input_len
                safe_max_new_tokens = max(1, remaining_context - 1)

                gen_kwargs = {
                    "max_new_tokens": safe_max_new_tokens,
                    "do_sample": False,
                    "pad_token_id": tokenizer.eos_token_id,
                    "eos_token_id": tokenizer.eos_token_id,
                }

                with torch.no_grad():
                    output = model.generate(**inputs, **gen_kwargs)

                generated_tokens = len(output[0]) - input_len
                generated_text = tokenizer.decode(
                    output[0][input_len:],
                    skip_special_tokens=True
                )

            except Exception as e:
                print(f"    ❌ Run {r+1} error: {e}", flush=True)

            finally:
                monitor.stop(token_count=generated_tokens)

            try:
                score, peak_temp, tps, avg_cpu, avg_gpu, duration, avg_power, energy_joules = monitor.get_score()
            except Exception as e:
                print(f"    ❌ get_score() failed run {r+1}: {type(e).__name__}: {e}", flush=True)
                continue

            print(f"      [run {r+1}/{runs}] score: {score:.2f} | TPS: {tps:.2f}", flush=True)

            # ✅ append AFTER the print, BEFORE the guard
            run_results.append({
                "score": score,
                "peak_temp": peak_temp,
                "tps": tps,
                "avg_gpu": avg_gpu,
                "duration": duration,
                "avg_power": avg_power,
                "energy_joules": energy_joules,
                "input_tokens": input_len,
                "output_tokens": generated_tokens,
                "generated_text": generated_text,
            })

        # ── Back in the for-i loop, AFTER all runs complete ──────────────
        if not run_results:
            print(f"  ⚠️ All runs failed for prompt {i+1}, skipping.", flush=True)
            continue

        def avg(key):
            return sum(entry[key] for entry in run_results) / len(run_results)

        scores.append({
            "prompt": prompt,
            "score": avg("score"),
            "peak_temp": avg("peak_temp"),
            "tps": avg("tps"),
            "avg_gpu": avg("avg_gpu"),
            "duration": avg("duration"),
            "avg_power": avg("avg_power"),
            "energy_joules": avg("energy_joules"),
            "input_tokens": run_results[0]["input_tokens"],
            "output_tokens": avg("output_tokens"),
            "output": run_results[-1].get("generated_text", "")
        })

        temp_str = f"{avg('peak_temp'):.1f}C"
        print(f"    --> Score: {avg('score'):.2f} | Temp: {temp_str} | TPS: {avg('tps'):.2f}", flush=True)
        if progress_callback:
            progress_callback({
                "status": "eval",
                "message": f"    --> Score: {avg('score'):.2f} | Temp: {temp_str} | TPS: {avg('tps'):.2f}"
            })

    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores


def run_sponge_attack(model_id, gens=5, pop=10, quantize=False, progress_callback=None):
    quant_mode = quantize if isinstance(quantize, str) else ("gguf-q4" if quantize else "none")
    quant_str = f" ({quant_mode})" if quant_mode != "none" else ""
    if progress_callback: progress_callback({"status": "starting", "message": f"Starting Sponge Attack GA on {model_id}{quant_str}"})
    print(f"Starting Sponge Attack GA on {model_id}{quant_str}")
    
    if progress_callback: progress_callback({"status": "loading", "message": f"Loading model {model_id}{quant_str}..."})
    print(f"Loading model {model_id}{quant_str}...")
    tokenizer, model, device, quant_label = load_model_and_tokenizer(model_id, quantize=quant_mode)
    
    # Initialize Population
    population = []
    for _ in range(pop):
        try:
            # Generate random tokens if possible
            p = generate_random_prompt(tokenizer, length=PROMPT_LENGTH) 
        except:
            p = generate_random_prompt(length=PROMPT_LENGTH)
        population.append(p)
    
    best_overall = None

    for gen in range(gens):
        if progress_callback: progress_callback({"status": "running", "message": f"Running Generation {gen + 1}/{gens}", "generation": gen + 1})
        print(f"\nGENERATION {gen + 1}")
        print("="*40)
        
        scored_pop = evaluate_population(population, model, tokenizer, device, progress_callback=progress_callback)
        
        # Log Best of Gen
        best_of_gen = scored_pop[0]
        if best_overall is None or best_of_gen["score"] > best_overall["score"]:
            best_overall = best_of_gen
        
        # Report progress
        if progress_callback:
            progress_callback({
                "status": "progress",
                "generation": gen + 1,
                "best_score": best_of_gen["score"],
                "best_temp": best_of_gen["peak_temp"],
                "best_prompt": best_of_gen["prompt"],
                "best_output": best_of_gen["output"],
                # CPU load removed as per user request
                "best_avg_gpu": best_of_gen.get("avg_gpu", 0),
                "best_duration": best_of_gen.get("duration", 0),
                "best_avg_power": best_of_gen.get("avg_power", 0),
                "best_energy_joules": best_of_gen.get("energy_joules", 0),
                "best_input_tokens": best_of_gen.get("input_tokens", 0),
                "best_output_tokens": best_of_gen.get("output_tokens", 0)
            })

        print(f"\nGeneration {gen+1} Winner:")
        print(f"   Prompt: '{best_of_gen['prompt']}'")
        print(f"   Score: {best_of_gen['score']:.2f} | Peak Temp: {best_of_gen['peak_temp']}C")
        
        # Selection (Keep Top 50%, deduplicated)
        top_half = scored_pop[:len(population)//5]
        seen = set()
        parents = []
        for p in top_half:
            if p["prompt"] not in seen:
                seen.add(p["prompt"])
                parents.append(p["prompt"])

        # If we don't have enough unique parents, fill with fresh random prompts
        while len(parents) < max(2, len(population) // 4):
            parents.append(generate_random_prompt(tokenizer, length=PROMPT_LENGTH))

        # New Population — always keep at least the best 1 (elitism)
        new_pop = [parents[0]]

        while len(new_pop) < pop:
            p1 = random.choice(parents)
            p2 = random.choice(parents)
            child = crossover(p1, p2, tokenizer)

            # ↑ Raise mutation rate and always mutate children from identical parents
            if p1 == p2 or random.random() < MUTATION_RATE:
                child = mutate(child, tokenizer)

            # Don't add exact duplicates if pool is still diverse enough
            if child not in new_pop or len(new_pop) >= pop - 2:
                new_pop.append(child)
            else:
                new_pop.append(generate_random_prompt(tokenizer, length=PROMPT_LENGTH))

        population = new_pop

    print("\n💀 Attack Search Complete.")
    if best_overall is not None:
        best_overall["quant_label"] = quant_label
    if progress_callback: progress_callback({"status": "complete", "result": best_overall})
    
    # Aggressively free model from VRAM/RAM (handles accelerate dispatch hooks)
    cleanup_model(model, tokenizer)
    model = None
    tokenizer = None
    import gc
    gc.collect()
    
    return best_overall

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", type=str, help="Model path/ID")
    parser.add_argument("--gens", type=int, default=5, help="Generations")
    parser.add_argument("--pop", type=int, default=10, help="Population size")
    
    args = parser.parse_args()
    
    run_sponge_attack(args.model_id, gens=args.gens, pop=args.pop)
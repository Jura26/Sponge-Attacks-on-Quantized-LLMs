import argparse
import sys
import io
import os
import time

# Force UTF-8 encoding for stdout/stderr to avoid charmap errors on Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding='utf-8')
    else:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding='utf-8')
    else:
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import psutil
import threading
import random
import string
import platform
import statistics

# Add backend to path so we can import the model loader
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))
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
MIN_NEW_TOKENS = 256
MAX_NEW_TOKENS = 1024  # Upper limit for token generation

class SystemMonitor:
    def __init__(self, device="cpu"):
        self.device = device
        self.running = False
        self.stats = {
            "temps": [],
            "cpu": [],
            "power": [],
            "gpu_load": [],
            "gpu_temp": []
        }
        self.start_time = 0
        self.end_time = 0
        self.token_count = 0
        self.thread = None

    def _get_temp(self):
        """Get the relevant temperature based on device (CPU or GPU)."""
        max_temp = 0
        try:
            if platform.system() == "Linux":
                temps = psutil.sensors_temperatures()
                if not temps: return 0
                for name, entries in temps.items():
                    for entry in entries:
                        current = getattr(entry, 'current', 0)
                        if current > max_temp:
                            max_temp = current
            elif platform.system() == "Windows":
                if self.device == "cuda":
                    # GPU temperature
                    try:
                        from hardware_monitor import get_gpu_stats
                        g_temp, _ = get_gpu_stats()
                        if g_temp > max_temp:
                            max_temp = g_temp
                    except Exception:
                        pass
                else:
                    # CPU temperature
                    try:
                        from hardware_monitor import get_cpu_stats
                        c_temp, _ = get_cpu_stats()
                        if c_temp > max_temp:
                            max_temp = c_temp
                    except Exception:
                        pass
                    # Fallback: ACPI thermal zones
                    if max_temp == 0:
                        try:
                            import wmi
                            import pythoncom
                            pythoncom.CoInitialize()
                            w = wmi.WMI(namespace="root\\cimv2")
                            for zone in w.Win32_PerfFormattedData_Counters_ThermalZoneInformation():
                                celsius = float(zone.Temperature) - 273.15
                                if celsius > max_temp:
                                    max_temp = celsius
                        except Exception:
                            pass
        except:
            pass
        return max_temp

    def _monitor_loop(self):
        while self.running:
            self.stats["temps"].append(self._get_temp())
            self.stats["cpu"].append(psutil.cpu_percent(interval=None))
            
            # Only fetch GPU stats when model is actually on CUDA
            if self.device == "cuda" and platform.system() == "Windows":
                try:
                    from hardware_monitor import get_gpu_stats, get_gpu_power
                    g_temp, g_load = get_gpu_stats()
                    self.stats["gpu_temp"].append(g_temp)
                    self.stats["gpu_load"].append(min(g_load, 100.0))
                    g_power = get_gpu_power()
                    if g_power > 0:
                        self.stats["power"].append(g_power)
                except:
                    pass
            
            time.sleep(0.1) 

    def start(self):
        self.running = True
        self.stats = {"temps": [], "cpu": [], "power": [], "gpu_load": [], "gpu_temp": []}
        self.start_time = time.time()
        self.token_count = 0
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.start()

    def stop(self, token_count=0):
        self.running = False
        self.end_time = time.time()
        self.token_count = token_count
        if self.thread:
            self.thread.join()
        
    def get_score(self):
        """Calculate fitness score aligned with Shumailov et al. (EuroS&P 2021).

        The paper's Black-box GA uses a single hardware measurement as fitness:
          - Primary:  energy consumption  E = P_avg × t  (Joules)
          - Fallback: inference latency   t  (seconds)   when power sensor unavailable

        This matches the paper's Black-box Energy GA and Black-box Time GA respectively.
        GPU load is NOT used as a fitness signal — it is reported separately as metadata.
        """
        if not self.stats["temps"] and not self.stats["cpu"]:
            return 0, 0, 0, 0, 0, 0, 0, 0
        
        avg_temp = statistics.mean(self.stats["temps"]) if self.stats["temps"] else 0
        max_temp = max(self.stats["temps"]) if self.stats["temps"] else 0
        avg_cpu = statistics.mean(self.stats["cpu"]) if self.stats["cpu"] else 0
        
        # GPU Stats (only populated when device is cuda)
        avg_gpu_load = min(statistics.mean(self.stats["gpu_load"]), 100.0) if self.stats.get("gpu_load") else 0
        avg_gpu_temp = statistics.mean(self.stats["gpu_temp"]) if self.stats.get("gpu_temp") else 0
        
        # Power draw (Watts)
        avg_power = statistics.mean(self.stats["power"]) if self.stats.get("power") else 0

        duration = self.end_time - self.start_time
        if duration <= 0: duration = 0.001
        
        tps = self.token_count / duration
        
        # Total energy consumed: E = P_avg × t  (Watt·seconds = Joules)
        # This is the paper's core energy formula E = (P_static + P_dynamic) × t
        energy_joules = avg_power * duration
        
        if energy_joules > 0:
            # Black-box Energy GA (paper §4.4.1): fitness = measured energy consumption
            score = energy_joules
        else:
            # Black-box Time GA fallback (paper §4.4.1): fitness = inference latency
            score = duration
        
        return score, max_temp, tps, avg_cpu, avg_gpu_load, duration, avg_power, energy_joules

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
    
    print(f"\nEvaluating {len(population)} prompts...")
    if progress_callback:
        progress_callback({"status": "eval", "message": f"Evaluating {len(population)} prompts..."})
    
    # Try to find model's max context length
    model_max_length = getattr(model.config, "max_position_embeddings", None)
    if model_max_length is None:
        model_max_length = getattr(model.config, "n_positions", None)
    
    # Fallback if unknown (common for some architectures)
    if model_max_length is None:
        model_max_length = 4096 # Safe modern default
        print(f"⚠️ Could not detect model context length. Defaulting to {model_max_length}.")

    for i, prompt in enumerate(population):
        # Cool down system before measurement to ensure fairness
        if progress_callback:
            progress_callback({"status": "eval", "message": f"  Cooling down before prompt {i+1}/{len(population)}..."})
        cooldown(target_temp=65, max_wait=5, device=device)
        
        print(f"  [{i+1}/{len(population)}] Testing: '{prompt[:30]}...'")
        if progress_callback:
            progress_callback({"status": "eval", "message": f"  [{i+1}/{len(population)}] Testing: '{prompt[:30]}...'"})
        
        monitor = SystemMonitor(device=device)
        monitor.start()

        generated_tokens = 0
        try:
            # --- RUN MODEL GENERATION ---
            # Truncate prompt to be safe (leave room for generation)
            max_input_len = model_max_length - 50 
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_len).to(device)
            input_len = inputs.input_ids.shape[1]
            
            # Calculate remaining space and ensure we generate at least something
            remaining_context = model_max_length - input_len
            safe_max_new_tokens = max(1, remaining_context - 1)
            # Cap the max new tokens so it doesn't run forever on models with huge context sizes
            safe_max_new_tokens = min(MAX_NEW_TOKENS, safe_max_new_tokens)
            
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=safe_max_new_tokens, 
                    do_sample=True,
                )
                generated_tokens = len(output[0]) - input_len
                generated_text = tokenizer.decode(output[0][input_len:], skip_special_tokens=True)
            # ---------------------------
        except Exception as e:
            print(f"    ❌ Error: {e}")
            generated_text = f"Error: {str(e)}"
        finally:
            monitor.stop(token_count=generated_tokens)
        
        score, peak_temp, tps, avg_cpu, avg_gpu, duration, avg_power, energy_joules = monitor.get_score()
        
        # Log appropriate load metric based on actual device
        if device == "cuda":
            load_msg = f"GPU: {avg_gpu:.1f}%"
            power_msg = f" | Power: {avg_power:.1f}W | Energy: {energy_joules:.1f}J" if avg_power > 0 else ""
        else:
            load_msg = f"CPU: {avg_cpu:.1f}%"
            power_msg = ""
            
        scores.append({
            "prompt": prompt,
            "score": score,
            "peak_temp": peak_temp,
            "tps": tps,
            "avg_cpu": avg_cpu,
            "avg_gpu": avg_gpu,
            "duration": duration,
            "avg_power": avg_power,
            "energy_joules": energy_joules,
            "input_tokens": input_len,
            "output_tokens": generated_tokens,
            "output": generated_text
        })
        temp_str = f"{peak_temp}C" if peak_temp > 0 else "N/A"
        print(f"    --> Score: {score:.2f} | Temp: {temp_str} | {load_msg}{power_msg} | TPS: {tps:.2f}")
        if progress_callback:
            progress_callback({
                "status": "eval", 
                "message": f"    --> Score: {score:.2f} | Temp: {temp_str} | {load_msg}{power_msg} | TPS: {tps:.2f}"
            })

    # Sort by score (descending)
    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores

def run_sponge_attack(model_id, gens=5, pop=10, progress_callback=None):
    if progress_callback: progress_callback({"status": "starting", "message": f"Starting Sponge Attack GA on {model_id}"})
    print(f"Starting Sponge Attack GA on {model_id}")
    
    if progress_callback: progress_callback({"status": "loading", "message": f"Loading model {model_id}..."})
    print(f"Loading model {model_id}...")
    tokenizer, model, device, quant_label = load_model_and_tokenizer(model_id)
    
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
                "best_avg_cpu": best_of_gen.get("avg_cpu", 0),
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
        
        # Selection (Keep Top 50%)
        top_half = scored_pop[:len(population)//2]
        parents = [p["prompt"] for p in top_half]
        
        # New Population
        new_pop = parents[:] # Elitism
        
        while len(new_pop) < pop:
            # Crossover
            p1 = random.choice(parents)
            p2 = random.choice(parents)
            child = crossover(p1, p2, tokenizer)
            
            # Mutation
            if random.random() < MUTATION_RATE:
                child = mutate(child, tokenizer)
            
            new_pop.append(child)
            
        population = new_pop

    print("\n💀 Attack Search Complete.")
    if best_overall is not None:
        best_overall["quant_label"] = quant_label
    if progress_callback: progress_callback({"status": "complete", "result": best_overall})
    
    # Aggressively free model from VRAM (handles accelerate dispatch hooks)
    cleanup_model(model, tokenizer)
    
    return best_overall

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", type=str, help="Model path/ID")
    parser.add_argument("--gens", type=int, default=5, help="Generations")
    parser.add_argument("--pop", type=int, default=10, help="Population size")
    
    args = parser.parse_args()
    
    run_sponge_attack(args.model_id, gens=args.gens, pop=args.pop)
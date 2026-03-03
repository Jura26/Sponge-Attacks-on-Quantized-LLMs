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
    from model import load_model_and_tokenizer
    import torch
except ImportError:
    print("❌ Could not import backend.model. Make sure you are running this from the project root.")
    sys.exit(1)

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
    def __init__(self):
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
        """Get the highest current temperature from any sensor."""
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
                # Primary: LibreHardwareMonitor .NET library
                try:
                    from hardware_monitor import get_max_temperature
                    t = get_max_temperature()
                    if t > max_temp:
                        max_temp = t
                except Exception:
                    # Fallback: ACPI thermal zones
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
            
            # Fetch GPU stats on Windows
            if platform.system() == "Windows":
                try:
                    from hardware_monitor import get_gpu_stats
                    g_temp, g_load = get_gpu_stats()
                    self.stats["gpu_temp"].append(g_temp)
                    self.stats["gpu_load"].append(g_load)
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
        """Calculate score based on Latency (Time) and Energy (Temp/Load)."""
        if not self.stats["temps"]: return 0, 0, 0, 0, 0, 0
        
        avg_temp = statistics.mean(self.stats["temps"])
        max_temp = max(self.stats["temps"])
        avg_cpu = statistics.mean(self.stats["cpu"]) if self.stats["cpu"] else 0
        
        # GPU Stats
        avg_gpu_load = statistics.mean(self.stats["gpu_load"]) if self.stats.get("gpu_load") else 0
        avg_gpu_temp = statistics.mean(self.stats["gpu_temp"]) if self.stats.get("gpu_temp") else 0

        duration = self.end_time - self.start_time
        if duration <= 0: duration = 0.001
        
        tps = self.token_count / duration
        
        # Determine if GPU was used
        is_gpu = False
        try:
            if torch.cuda.is_available() or avg_gpu_load > 10:
                is_gpu = True
        except:
            pass

        if is_gpu:
            # GPU Scoring Strategy
            latency_factor = 0
            if tps > 0:
                latency_factor = 100 / tps
            
            # Weighted sum
            score = (avg_gpu_load * 1.0) + \
                    (avg_gpu_temp * 1.0) + \
                    (self.token_count * 0.1) + \
                    (latency_factor * 1.0)
        else:
            # CPU Fallback
            score = (avg_temp * 0.5) + (duration * 20)
        
        return score, max_temp, tps, avg_cpu, avg_gpu_load, duration

def cooldown(target_temp=60, max_wait=10):
    """Wait for CPU to cool down to ensure fair testing."""
    print(f"  Cooling down (target < {target_temp}C)...", end="", flush=True)
    temp_monitor = SystemMonitor()
    
    for _ in range(max_wait):
        current_temp = temp_monitor._get_temp()
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
        cooldown(target_temp=65, max_wait=5)
        
        print(f"  [{i+1}/{len(population)}] Testing: '{prompt[:30]}...'")
        if progress_callback:
            progress_callback({"status": "eval", "message": f"  [{i+1}/{len(population)}] Testing: '{prompt[:30]}...'"})
        
        monitor = SystemMonitor()
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
        
        score, peak_temp, tps, avg_cpu, avg_gpu, duration = monitor.get_score()
        
        # Log appropriate load metric
        load_msg = f"CPU: {avg_cpu:.1f}%"
        if avg_gpu > 0:
            load_msg = f"GPU: {avg_gpu:.1f}%"
            
        scores.append({
            "prompt": prompt,
            "score": score,
            "peak_temp": peak_temp,
            "tps": tps,
            "avg_cpu": avg_cpu,
            "avg_gpu": avg_gpu,
            "duration": duration,
            "input_tokens": input_len,
            "output_tokens": generated_tokens,
            "output": generated_text
        })
        print(f"    --> Score: {score:.2f} | Temp: {peak_temp}C | {load_msg} | TPS: {tps:.2f}")
        if progress_callback:
            progress_callback({
                "status": "eval", 
                "message": f"    --> Score: {score:.2f} | Temp: {peak_temp}C | {load_msg} | TPS: {tps:.2f}"
            })

    # Sort by score (descending)
    scores.sort(key=lambda x: x["score"], reverse=True)
    return scores

def run_sponge_attack(model_id, gens=5, pop=10, progress_callback=None):
    if progress_callback: progress_callback({"status": "starting", "message": f"Starting Sponge Attack GA on {model_id}"})
    print(f"Starting Sponge Attack GA on {model_id}")
    
    if progress_callback: progress_callback({"status": "loading", "message": "Loading model..."})
    print("Loading model...")
    tokenizer, model, device = load_model_and_tokenizer(model_id)
    
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
                "best_input_tokens": best_of_gen.get("input_tokens", 0),
                "best_output_tokens": best_of_gen.get("output_tokens", 0)
            })

        print(f"\nGeneration {gen+1} Winner:")
        print(f"   Prompt: '{best_of_gen['prompt']}'")
        print(f"   Score: {best_of_gen['score']:.2f} | Peak Temp: {best_of_gen['peak_temp']}C")
        
        # Write to log file
        with open("sponge_log.txt", "a") as f:
            f.write(f"Gen {gen+1}: {best_of_gen['score']:.2f} | {best_of_gen['peak_temp']}C | {best_of_gen['prompt']}\n")
            
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
    if progress_callback: progress_callback({"status": "complete", "result": best_overall})
    return best_overall

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id", type=str, help="Model path/ID")
    parser.add_argument("--gens", type=int, default=5, help="Generations")
    parser.add_argument("--pop", type=int, default=10, help="Population size")
    
    args = parser.parse_args()
    
    run_sponge_attack(args.model_id, gens=args.gens, pop=args.pop)
# Book Scanning - Instance Generator & ILS Solver

This repository contains tools for generating custom, heterogeneous instances for the Google Hash Code 2020 Book Scanning problem, as well as a high-performance Iterated Local Search (ILS) solver. These tools are designed to evaluate and stress-test algorithms for a master's thesis.

---

## 1. Instance Generator (`generator.py`)

The generator uses the original Hash Code 2020 instances (e.g., `b_read_on.txt`, `c_incunabula.txt`) as "seeds" to create new instances. It uses **NumPy** for vectorized operations, making it extremely fast (able to generate massive instances in less than a second).

### How it works
The generator takes a seed file, a `scale` factor, and a `noise` (perturbation) factor. It modifies the seed by:
1. **Scaling overall dimensions:** Multiplies the total number of Books ($B$), Libraries ($L$), and Days ($D$) by the scale factor, then applies $\pm$ random noise so every instance is uniquely sized.
2. **Perturbing scores:** For uniform-score seeds (b, d), samples from a normal distribution centered at the original mean. For heterogeneous seeds (c, e, f), resamples from the original distribution with multiplicative noise. This preserves each seed's statistical character without artificial periodicity.
3. **Popularity-weighted book assignment:** Generates exponential popularity weights for books, then uses weighted sampling so "popular" books naturally appear in many libraries, creating realistic overlap patterns.
4. **Feasibility guarantee:** Ensures the generated deadline $D$ is always greater than the minimum library signup time, preventing infeasible instances.
5. **Instance validation:** Every generated instance is validated (correct dimensions, book indices in range, no duplicate books, positive signup/ship-rate values) before writing to disk.

### Reproducibility

All randomness is seeded deterministically using SHA-256 hashing of the parameter tuple `(random_seed, seed_letter, scale, tightness, replicate)`. This guarantees **identical output across Python versions, platforms, and runs** (unlike Python's built-in `hash()` which is randomized since Python 3.3).

To reproduce the default dataset:
```bash
python generator.py --batch --seed 42 --out_dir instances
```

### Single-Instance Mode

Run the generator from the terminal via:
```bash
python generator.py seed/b_read_on.txt --count 5 --scale 0.8 --noise 0.2
```

**Parameters:**
- `seed_file` (Required): The path to the original text file used as the base (e.g., `seed/d_tough_choices.txt`).
- `--count` (Optional): The number of new instances to generate (default: 1).
- `--scale` (Optional): Factor to increase/decrease the problem size. For example, `1.5` makes the instance roughly 50% larger; `0.8` makes it 20% smaller (default: 1.0).
- `--noise` (Optional): Fluctuation percentage applied to metrics. `0.2` means properties will randomly fluctuate by $\pm 20\%$ (default: 0.1).
- `--tightness` (Optional): Controls $D$ relative to total signup time. Range 0.1 (hard/constrained) to 1.0 (easy/loose). When omitted, $D$ scales from the seed as before.
- `--out_dir` (Optional): Directory where generated files are saved (default: `generated_instances`).
- `--seed` (Optional): Set a random seed for reproducible generation.

### Batch Mode

Batch mode generates a systematic full-factorial grid of instances with a single command. The grid dimensions are fully configurable:

```bash
# Default: 5 seeds x 5 scales x 5 tightness = 125 instances
python generator.py --batch --seed 42 --out_dir instances

# With replicates for statistical significance: 5x5x5x3 = 375 instances
python generator.py --batch --seed 42 --replicates 3 --out_dir instances

# Custom grid: 5 seeds x 6 scales x 5 tightness = 150 instances
python generator.py --batch --seed 42 --scales 0.25 0.5 0.75 1.0 1.5 2.0 --out_dir instances

# Include original Hash Code seed instances in output
python generator.py --batch --seed 42 --include_seeds --out_dir instances
```

**Default parameter grid (125 instances):**

| Scale | Books (approx.) | Tightness levels |
|-------|-----------------|------------------|
| 0.25 | ~25K | 0.1, 0.25, 0.5, 0.75, 1.0 |
| 0.50 | ~50K | 0.1, 0.25, 0.5, 0.75, 1.0 |
| 0.75 | ~75K | 0.1, 0.25, 0.5, 0.75, 1.0 |
| 1.00 | ~100K | 0.1, 0.25, 0.5, 0.75, 1.0 |
| 1.50 | ~150K | 0.1, 0.25, 0.5, 0.75, 1.0 |

All batch instances use `noise=0.2` and a fixed random seed (default 42) for reproducibility.

**Naming convention:** `{seed_letter}_{scale}x_{tightness}t[_r{N}].txt`
- Without replicates: `b_025x_010t.txt` — seed b, scale 0.25, tightness 0.1
- With replicates: `b_025x_010t_r1.txt`, `b_025x_010t_r2.txt`, ...
- Regex-parseable: `([bcdef])_(\d+)x_(\d+)t(?:_r(\d+))?\.txt`

**Batch parameters:**
- `--batch` (Required): Enables batch mode.
- `--seed_dir` (Optional): Directory containing seed files (default: `seed`).
- `--seed` (Optional): Random seed for reproducibility (default: 42).
- `--out_dir` (Optional): Output directory (default: `generated_instances`).
- `--scales` (Optional): Custom list of scale factors (default: `0.25 0.5 0.75 1.0 1.5`).
- `--tightness_levels` (Optional): Custom list of tightness levels (default: `0.1 0.25 0.5 0.75 1.0`).
- `--replicates` (Optional): Number of replicates per (seed, scale, tightness) combination (default: 1). Use 3-5 for statistical significance when benchmarking solvers.
- `--include_seeds` (Optional): Copy the original Hash Code seed instances into the output directory and include their statistics in `summary.csv`.
- `--noise` (Optional): Noise factor for batch mode (default: 0.2).

**Output:** A `summary.csv` file is written to the output directory with per-instance statistics including: B, L, D, total signup, actual tightness, signup/ship-rate/score statistics (mean, std, CV), sampled Jaccard overlap, book coverage, and replicate number (when replicates > 1).

---

## 2. ILS Solver (`final_ils_solver.py`)

The solver uses Iterated Local Search combined with an Adaptive Heap greedy initialization strategy. It is heavily optimized, relying on **in-place** modifications (zero deep copying) and **Numba JIT** compiled code for $50\times$ faster scoring evaluations.

### Usage

To run the solver, provide the input file, output file name, and search time limit (in seconds):
```bash
python final_ils_solver.py generated_instances/d_tough_choices_s0.8_n0.2_1.txt my_solution.txt 60
```

**Required Positional Arguments:**
1. `input_path`: Path to the compiled instance you want to solve (e.g., `generated_instances/...`).
2. `output_path`: Path & filename where the final submission format should be saved (e.g., `solution.txt`).
3. `time_limit`: Number of seconds the local search should run (e.g., `30` or `60`).

**Optional Tuning Parameters:**
You can pass additional flags to adjust the algorithm's behavior:
- `--p_swap`: Probability of choosing a "SWAP" move (default: 0.4).
- `--p_insert`: Probability of choosing an "INSERT/REMOVE" move (default: 0.3). The implicit remainder goes to "MOVE" operations.
- `--anneal_prob`: Probability of accepting a worse solution to escape local optima via Simulated Annealing (default: 0.0005).
- `--alphas`: Provide a list of alpha values to use in the Adaptive Heap initialization (default: `0.5 1.0 1.5 2.0 3.0`).
- `--run_exact`: Add this flag to also run the exact $O(N^2)$ weighted initialization (slower but potentially better starting bounds).
- `--seed`: Random seed for reproducibility.

**Example with custom heuristics:**
```bash
python final_ils_solver.py my_instance.txt sol.txt 120 --p_swap 0.5 --anneal_prob 0.001 --alphas 1.0 2.0
```

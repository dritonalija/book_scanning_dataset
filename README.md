# Book Scanning - Instance Generator & ILS Solver

This repository contains tools for generating custom, heterogeneous instances for the Google Hash Code 2020 Book Scanning problem, as well as a high-performance Iterated Local Search (ILS) solver. These tools are designed to evaluate and stress-test algorithms for a master's thesis.

---

## 1. Instance Generator (`generator.py`)

The generator uses the original Hash Code 2020 instances (e.g., `b_read_on.txt`, `c_incunabula.txt`) as "seeds" to create new instances. It uses **NumPy** for vectorized operations, making it extremely fast (able to generate massive instances in less than a second).

### How it works
The generator takes a seed file, a `scale` factor, and a `noise` (perturbation) factor. It modifies the seed by:
1. **Scaling overall dimensions:** Multiplies the total number of Books ($B$), Libraries ($L$), and Days ($D$) by the scale factor, then applies $\pm$ random noise so every instance is uniquely sized.
2. **Perturbing scores:** Copies the original book score distribution and injects random noise.
3. **Perturbing libraries:** Randomly alters library signup times, shipping rates, and the number of books originally assigned to them.
4. **Randomized book assignment:** Efficiently selects random books to assign to libraries matching the new scale.

### Usage

Run the generator from the terminal via:
```bash
python generator.py seed_instance.txt --count 5 --scale 0.8 --noise 0.2
```

**Parameters:**
- `seed_instance.txt` (Required): The path to the original text file used as the base (e.g., `d_tough_choices.txt`).
- `--count` (Optional): The number of new instances to generate (default: 1).
- `--scale` (Optional): Factor to increase/decrease the problem size. For example, `1.5` makes the instance roughly 50% larger; `0.8` makes it 20% smaller (default: 1.0).
- `--noise` (Optional): Fluctuation percentage applied to metrics. `0.2` means properties will randomly fluctuate by $\pm 20\%$ (default: 0.1).
- `--out_dir` (Optional): Directory where generated files are saved (default: `generated_instances`).
- `--seed` (Optional): Set a random seed for reproducible generation.

*Note: For Instance B (`b_read_on.txt`), because it is perfectly uniform, you need a high noise factor (e.g., `0.3` to `0.5`) to make the generated instances truly heterogeneous.*

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

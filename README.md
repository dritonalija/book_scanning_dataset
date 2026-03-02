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

All randomness is seeded deterministically using SHA-256 hashing of the parameter tuple `(random_seed, seed_letter, scale, tightness)`. This guarantees **identical output across Python versions, platforms, and runs** (unlike Python's built-in `hash()` which is randomized since Python 3.3).

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
- `--noise` (Optional): Fluctuation percentage applied to metrics. `0.2` means properties will randomly fluctuate by $\pm 20\%$ (default: 0.2).
- `--tightness` (Optional): Controls $D$ relative to total signup time. Range 0.1 (heavily constrained - only ~10% of libraries can sign up) to 1.0 (least constrained - enough time to sign up all libraries sequentially). When omitted, $D$ scales from the seed as before.
- `--out_dir` (Optional): Directory where generated files are saved (default: `generated_instances`).
- `--seed` (Optional): Set a random seed for reproducible generation.

### Small Example (Scale + Tightness + Noise)

Assume a seed with `B=1000`, `L=100`, and one library with `signup=10`, `ship_rate=4`, `n_books=120`.

If you run with `--scale 0.5 --tightness 0.5 --noise 0.2`, one possible generated outcome is:
- `B` around `1000 * 0.5`, then perturbed in `[0.8, 1.2]` (for example, `550`)
- `L` around `100 * 0.5`, then perturbed in `[0.8, 1.2]` (for example, `45`)
- library `signup` around `10`, perturbed in `[0.8, 1.2]` (for example, `12`)
- library `ship_rate` around `4`, perturbed in `[0.8, 1.2]` (for example, `3`)
- library `n_books` around `120 * 0.5`, perturbed in `[0.8, 1.2]` (for example, `66`)
- final `D` is derived from `tightness * total_signup` of generated libraries

### Batch Mode

Batch mode generates a systematic full-factorial grid of instances with a single command. The grid dimensions are fully configurable:

```bash
# Default: 5 seeds x 4 scales x 5 tightness = 100 instances
python generator.py --batch --seed 42 --out_dir instances

# Include original Hash Code seed instances in output
python generator.py --batch --seed 42 --include_seeds --out_dir instances
```

**Default parameter grid (100 instances):**

| Scale | Books (approx.) | Tightness levels |
|-------|-----------------|------------------|
| 0.25 | ~25K | 0.1, 0.25, 0.5, 0.75, 1.0 |
| 0.50 | ~50K | 0.1, 0.25, 0.5, 0.75, 1.0 |
| 0.75 | ~75K | 0.1, 0.25, 0.5, 0.75, 1.0 |
| 1.00 | ~100K | 0.1, 0.25, 0.5, 0.75, 1.0 |

All batch instances use `noise=0.2` and a fixed random seed (default 42) for reproducibility.

**Naming convention:** `{seed_letter}_{scale}x_{tightness}t.txt`
- Example: `b_025x_010t.txt` — seed b, scale 0.25, tightness 0.1
- Regex-parseable: `([bcdef])_(\d+)x_(\d+)t\.txt`

**Batch parameters:**
- `--batch` (Required): Enables batch mode.
- `--seed_dir` (Optional): Directory containing seed files (default: `seed`).
- `--seed` (Optional): Random seed for reproducibility (default: 42).
- `--out_dir` (Optional): Output directory (default: `generated_instances`).
- `--scales` (Optional): Custom list of scale factors (default: `0.25 0.5 0.75 1.0`; with `--crossbreed`: `0.25 0.5 1.0`).
- `--tightness_levels` (Optional): Custom list of tightness levels (default: `0.1 0.25 0.5 0.75 1.0`; with `--crossbreed`: `0.1 0.5 1.0`).
- Values that round to the same `x100` filename token are rejected in batch mode (for example `0.251` and `0.252` both map to `025`).
- `--include_seeds` (Optional): Copy the original Hash Code seed instances into the output directory and include their statistics in `summary.csv`.
- `--noise` (Optional): Noise factor for batch mode (default: 0.2).

**Output:** A `summary.csv` file is written to the output directory with per-instance statistics including: B, L, D, total signup, actual tightness, signup/ship-rate/score statistics (mean, std, variance, CV), sampled Jaccard overlap, book coverage, and book duplication rate.

### Cross-Breeding Mode

Cross-breeding generates hybrid instances by combining characteristics from **two different seed instances**: book scores from one seed and library structure (signup times, ship rates, collection sizes) from another. This fills gaps in the feature space that single-seed generation cannot reach.

With `--crossbreed`, the 5 single seeds are joined by $\binom{5}{2} = 10$ cross-bred pairs, giving **15 sources** total. All sources share the same scale x tightness grid, so you control the total instance count by adjusting the grid:

```bash
# Default crossbreed: 15 sources x 3 scales x 3 tightness = 135 instances
python generator.py --batch --seed 42 --crossbreed --out_dir instances

# Custom grid
python generator.py --batch --seed 42 --crossbreed --scales 0.5 1.0 --tightness_levels 0.1 0.25 0.5 1.0 --out_dir instances
```

**Naming convention for hybrids:** `{letter_a}{letter_b}_{scale}x_{tightness}t.txt`
- Example: `bc_050x_025t.txt` — scores from seed b, library structure from seed c, scale 0.50, tightness 0.25

**Parameters:**
- `--crossbreed` (Optional): Enable cross-breeding. Only works with `--batch` and requires at least 2 seed files.

---

## 2. Feature Analysis (`analyze.py`)

After generating instances, use `analyze.py` to visualize the feature space and verify that your instances have good diversity. It performs PCA dimensionality reduction on 11 instance features and produces an interactive Plotly scatter plot.

```bash
# Basic: open interactive plot in browser
python analyze.py --summary instances/summary.csv

# Save plot as HTML file
python analyze.py --summary instances/summary.csv --save feature_space.html
```

**Features used for PCA:** B, L, D, score mean, score variance, signup mean, ship rate mean, book duplication rate, book coverage, library size mean, actual tightness.

**Dependencies:** `pandas`, `scikit-learn`, `plotly`

---

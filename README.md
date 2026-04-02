# Book Scanning Dataset Generator

This repository contains a configurable instance generator for the Google Hash Code 2020 Book Scanning problem.

The generator builds new problem instances from the original Hash Code seed files and is designed to produce realistic, diverse, and reproducible benchmarks for testing optimization algorithms.

## Repository Structure

- `generator.py`: dataset generator
- `analyze.py`: feature-space visualization and diversity analysis
- `seed/`: original Hash Code 2020 seed instances
- `instances/`: example 135-instance dataset generated with the final settings

## Requirements

Python 3.10+ is recommended.

Core dependencies:

```bash
pip install numpy
```

Optional dependencies:

```bash
pip install pandas scikit-learn plotly numba
```

- `pandas`, `scikit-learn`, and `plotly` are only needed for `analyze.py`

## Example Dataset

`instances/` is an example 135-instance dataset generated with the final settings.

It is built from:

- 5 seed families: `b, c, d, e, f`
- 10 cross-bred families: `bc, bd, be, bf, cd, ce, cf, de, df, ef`
- 3 scale levels
- 3 tightness levels

Naming convention:

- single-seed instance: `b_060x_008t.txt`
- hybrid instance: `ce_075x_012t.txt`

Interpretation:

- `ce`: scores from seed `c`, structure from seed `e`
- `060x`: scale `0.60`
- `008t`: tightness `0.08`

This gives `15 x 3 x 3 = 135` instances.

## Generator

`generator.py` is the main generator in this repository.

It preserves seed-family structure while introducing controlled variation in:

- instance size
- signup times
- ship rates
- library sizes
- score distributions
- book overlap patterns
- tightness

The current version also includes mild structure-aware adjustments for difficult seed families such as `b`, `c`, and `d`, so the generated instances remain faithful to the original seeds while being more informative for benchmarking.

### How It Works

Starting from one seed instance, the generator creates a new instance by modifying:

1. `B` (number of books)
2. `L` (number of libraries)
3. `D` (number of days)
4. book scores
5. library signup times
6. library ship rates
7. library book lists

The main mechanisms are:

1. Size scaling: `B` and `L` are scaled by the requested `scale` factor and then perturbed by bounded random noise.
2. Score generation: new book scores are sampled from the empirical seed score distribution, preserving the seed family's statistical character.
3. Structure transfer: generated libraries inherit a noisy version of the seed library structure.
4. Overlap construction: books are assigned with popularity-based sampling, template anchoring, and controlled overlap so instances stay realistic.
5. Tightness control: `D` is either scaled from the seed or derived from the requested `tightness` target.
6. Validation: every generated instance is checked against the official Hash Code bounds.

### Interpreting Generated Instances

The most important generation controls are `scale` and `tightness`.

- `scale` controls the approximate instance size
- `tightness` controls the deadline pressure relative to total signup time

In practical terms:

- larger `scale` usually means more books, more libraries, and larger libraries
- lower `tightness` usually means fewer available days relative to signup cost, so the instance is harder
- higher `tightness` usually means more generous deadlines, so more libraries can contribute

For a generated library:

- `n_books` is the number of books listed in that library
- `signup` is the number of days needed before the library can start shipping
- `ship_rate` is the number of books that can be scanned per day after signup

`n_books` does not just copy the seed directly. It is scaled from the corresponding seed library, perturbed with noise, and then capped to remain valid.

### Small Example

Assume a seed instance has:

- `B = 1000`
- `L = 100`
- one library with `signup = 10`, `ship_rate = 4`, `n_books = 120`

If you generate with:

```bash
python generator.py seed/b_read_on.txt --count 1 --scale 0.5 --tightness 0.12 --noise 0.2 --seed 42
```

then one possible generated outcome is:

- `B` around `1000 x 0.5`, then perturbed in a bounded range
- `L` around `100 x 0.5`, then perturbed in a bounded range
- library `n_books` around `120 x 0.5`, then perturbed and capped
- library `signup` around `10`, with bounded perturbation
- library `ship_rate` around `4`, with bounded perturbation
- final `D` derived from the requested `tightness` target and the generated total signup

So `scale` mainly changes the size of the instance, while `tightness` changes how much scheduling pressure the solver faces.

### Reproducibility

Generation is deterministic.

Each candidate instance uses a SHA-256-based seed derived from:

- `random_seed`
- source name
- scale
- tightness
- attempt number

So the same command with the same code and the same seed files produces the same dataset.

### Generate The Example 135-Instance Dataset

To reproduce the example 135-instance dataset:

```bash
python generator.py --batch --seed_dir seed --crossbreed --seed 42 --out_dir instances --scales 0.6 0.75 0.9 --tightness_levels 0.08 0.12 0.2
```

This produces:

- `135` instance files
- `summary.csv`
- `generation_config.json`

### Single-Instance Mode

Example:

```bash
python generator.py seed/b_read_on.txt --count 3 --scale 0.75 --tightness 0.12 --noise 0.2 --seed 42 --out_dir generated_instances
```

### Batch Mode

Example:

```bash
python generator.py --batch --seed_dir seed --crossbreed --seed 42 --out_dir instances_custom --scales 0.6 0.75 0.9 --tightness_levels 0.08 0.12 0.2
```

Main batch arguments:

- `--batch`: enable grid generation
- `--seed_dir`: directory containing seed `.txt` files
- `--crossbreed`: include hybrid score/structure combinations
- `--seed`: random seed for deterministic generation
- `--out_dir`: output directory
- `--scales`: scale grid
- `--tightness_levels`: tightness grid
- `--noise`: perturbation level, default `0.2`
- `--include_seeds`: also copy original seed instances into output

## Feature Analysis

`analyze.py` provides a compact view of dataset diversity by projecting instance-level features into a lower-dimensional feature space.

It is intended to help inspect:

- how different generated instances are from one another
- whether seed families occupy distinct regions of the feature space
- whether the generated dataset covers a broad range of structural characteristics

Typical features include:

- problem size
- tightness
- score statistics
- signup and shipping statistics
- overlap and duplication statistics

Example usage:

```bash
python analyze.py --summary instances/summary.csv
python analyze.py --summary instances/summary.csv --save feature_space.html
```

The output can be used to visually assess whether the generator is producing a diverse and well-spread collection of instances.

## Notes

- `generator.py` is the main generator in this repository.
- `instances/` is an example dataset produced with the final generation settings.
- The generator is deterministic as long as the code, seed files, and command-line parameters remain unchanged.

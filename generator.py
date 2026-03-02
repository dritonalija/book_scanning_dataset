import sys
import argparse
import os
import csv
import glob
import hashlib
import shutil
import numpy as np
import time


# Seeds with uniform score distributions (need normal sampling)
UNIFORM_SCORE_SEEDS = {'b', 'd'}

# Hash Code 2020 official bounds (from problem statement)
MAX_B = 10**5       # max books
MAX_L = 10**5       # max libraries
MAX_D = 10**5       # max days
MAX_SCORE = 10**3   # max score per book (0 <= S_i <= 1000)
MAX_N = 10**5       # max books per library
MAX_T = 10**5       # max signup days per library
MAX_M = 10**5       # max ship rate per library
MAX_TOTAL_BOOKS = 10**6  # total books across all libraries


def read_instance(filepath):
    """Reads a HashCode 2020 Book Scanning problem instance."""
    with open(filepath, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    B, L, D = map(int, lines[0].split())
    scores = list(map(int, lines[1].split()))

    libraries = []
    idx = 2
    for _ in range(L):
        n_books, signup, ship_rate = map(int, lines[idx].split())
        books_in_lib = list(map(int, lines[idx + 1].split()))
        libraries.append({
            'n_books': n_books,
            'signup': signup,
            'ship_rate': ship_rate,
            'books': books_in_lib
        })
        idx += 2

    return B, L, D, scores, libraries


def write_instance(filepath, B, L, D, scores, libraries):
    """Writes the generated instance to a file."""
    with open(filepath, 'w') as f:
        f.write(f"{B} {L} {D}\n")
        f.write(" ".join(map(str, scores)) + "\n")
        for lib in libraries:
            f.write(f"{lib['n_books']} {lib['signup']} {lib['ship_rate']}\n")
            f.write(" ".join(map(str, lib['books'])) + "\n")


def validate_instance(B, L, D, scores, libraries):
    """Validate that a generated instance conforms to Hash Code 2020 bounds."""
    errors = []
    # Global bounds
    if not (1 <= B <= MAX_B):
        errors.append(f"B={B} out of range [1, {MAX_B}]")
    if not (1 <= L <= MAX_L):
        errors.append(f"L={L} out of range [1, {MAX_L}]")
    if not (1 <= D <= MAX_D):
        errors.append(f"D={D} out of range [1, {MAX_D}]")
    if len(scores) != B:
        errors.append(f"Score count {len(scores)} != B={B}")
    if len(libraries) != L:
        errors.append(f"Library count {len(libraries)} != L={L}")

    # Score bounds
    for idx, s in enumerate(scores):
        if not (0 <= s <= MAX_SCORE):
            errors.append(f"Score[{idx}]={s} out of range [0, {MAX_SCORE}]")
            break

    # Per-library and total books
    total_books = 0
    for i, lib in enumerate(libraries):
        if lib['n_books'] != len(lib['books']):
            errors.append(f"Library {i}: n_books={lib['n_books']} != len(books)={len(lib['books'])}")
        if not (1 <= lib['signup'] <= MAX_T):
            errors.append(f"Library {i}: signup={lib['signup']} out of range [1, {MAX_T}]")
        if not (1 <= lib['ship_rate'] <= MAX_M):
            errors.append(f"Library {i}: ship_rate={lib['ship_rate']} out of range [1, {MAX_M}]")
        if not (1 <= lib['n_books'] <= MAX_N):
            errors.append(f"Library {i}: n_books={lib['n_books']} out of range [1, {MAX_N}]")
        for book_id in lib['books']:
            if book_id < 0 or book_id >= B:
                errors.append(f"Library {i}: book {book_id} out of range [0, {B})")
                break
        if len(lib['books']) != len(set(lib['books'])):
            errors.append(f"Library {i}: duplicate books")
        total_books += lib['n_books']

    if total_books > MAX_TOTAL_BOOKS:
        errors.append(f"Total books across libraries {total_books} > {MAX_TOTAL_BOOKS}")

    if errors:
        raise ValueError("Instance validation failed:\n" + "\n".join(errors))
    return True


def detect_seed_letter(filepath):
    """Extract the seed letter (b, c, d, e, f) from a seed filename."""
    basename = os.path.basename(filepath).lower()
    if basename and basename[0] in 'bcdef':
        return basename[0]
    return None


def generate_scores(orig_scores, new_B, seed_letter, noise, rng):
    """Generate book scores preserving the seed's statistical character.

    For uniform-score seeds (b, d): sample from a normal distribution
    centered at the original score mean, with spread controlled by noise.

    For heterogeneous seeds (c, e, f): resample from original scores
    with replacement, then apply multiplicative noise.
    """
    orig = np.array(orig_scores, dtype=np.float64)
    mean_score = orig.mean()

    if seed_letter in UNIFORM_SCORE_SEEDS:
        # Normal distribution centered at original mean
        spread = max(mean_score * noise, 1.0)
        new_scores = rng.normal(loc=mean_score, scale=spread, size=new_B)
    else:
        # Resample from original distribution with replacement, then perturb
        resampled = rng.choice(orig, size=new_B, replace=True)
        noise_vals = rng.uniform(1.0 - noise, 1.0 + noise, size=new_B)
        new_scores = resampled * noise_vals

    # Clamp to [0, MAX_SCORE] per Hash Code spec and convert to int
    new_scores = np.clip(np.round(new_scores), 0, MAX_SCORE).astype(np.int64)
    return new_scores.tolist()


def assign_books_with_overlap(new_B, target_n_books, rng, popularity_weights):
    """Assign books to a library using popularity-weighted sampling.

    Popular books appear in many libraries naturally, creating realistic
    overlap patterns without expensive pairwise computation.
    """
    target_n_books = min(target_n_books, new_B)
    # Weighted sampling without replacement
    books = rng.choice(new_B, size=target_n_books, replace=False, p=popularity_weights)
    return books.tolist()


def generate_popularity_weights(new_B, rng):
    """Generate popularity weights for books using an exponential distribution.

    A small number of books will be very popular (high weight) while most
    books have moderate popularity, creating natural overlap when libraries
    sample from this distribution.
    """
    raw_weights = rng.exponential(scale=1.0, size=new_B)
    weights = raw_weights / raw_weights.sum()
    return weights


def generate_new_instance(B, L, D, scores, libraries, scale_factor, noise, rng,
                          tightness=None, seed_letter=None):
    """Generates a new instance by scaling and perturbing the seed instance."""
    # Apply noise to B and L dimensions, clamped to Hash Code bounds
    new_B = max(1, min(int(B * scale_factor * rng.uniform(1.0 - noise, 1.0 + noise)), MAX_B))
    new_L = max(1, min(int(L * scale_factor * rng.uniform(1.0 - noise, 1.0 + noise)), MAX_L))

    # Generate scores using improved method
    if seed_letter is None:
        seed_letter = 'c'  # default to heterogeneous behavior
    new_scores = generate_scores(scores, new_B, seed_letter, noise, rng)

    # Generate popularity weights for book assignment
    popularity_weights = generate_popularity_weights(new_B, rng)

    # Map each new library to a seed library:
    # - same size: 1-to-1 mapping
    # - downscale (new_L < L): sample without replacement (no duplicates)
    # - upscale (new_L > L): use all L at least once, then fill remainder randomly
    if new_L == L:
        lib_indices = list(range(L))
    elif new_L < L:
        lib_indices = rng.choice(L, size=new_L, replace=False).tolist()
    else:
        base = np.arange(L)
        extra = rng.choice(L, size=new_L - L, replace=True)
        lib_indices = np.concatenate([base, extra])
        rng.shuffle(lib_indices)
        lib_indices = lib_indices.tolist()

    new_libraries = []
    total_signup = 0
    for i in range(new_L):
        orig_lib = libraries[lib_indices[i]]

        # Perturb signup and ship rate, clamped to Hash Code bounds
        new_signup = max(1, min(int(orig_lib['signup'] * rng.uniform(1.0 - noise, 1.0 + noise)), MAX_T))
        new_ship_rate = max(1, min(int(orig_lib['ship_rate'] * rng.uniform(1.0 - noise, 1.0 + noise)), MAX_M))

        # Determine target number of books (clamped to new_B and MAX_N)
        target_n_books = max(1, min(
            int(orig_lib['n_books'] * scale_factor * rng.uniform(1.0 - noise, 1.0 + noise)),
            new_B, MAX_N
        ))

        # Popularity-weighted book assignment
        new_books = assign_books_with_overlap(new_B, target_n_books, rng, popularity_weights)

        new_libraries.append({
            'n_books': len(new_books),
            'signup': new_signup,
            'ship_rate': new_ship_rate,
            'books': new_books
        })
        total_signup += new_signup

    # Enforce total books across all libraries <= MAX_TOTAL_BOOKS
    total_books = sum(lib['n_books'] for lib in new_libraries)
    if total_books > MAX_TOTAL_BOOKS:
        # Trim libraries from the end until under budget
        while total_books > MAX_TOTAL_BOOKS and len(new_libraries) > 1:
            removed = new_libraries.pop()
            total_books -= removed['n_books']
            total_signup -= removed['signup']
        new_L = len(new_libraries)

    # Compute D based on tightness or scale from seed, clamped to MAX_D
    if tightness is not None:
        new_D = max(1, min(int(total_signup * tightness), MAX_D))
    else:
        new_D = max(1, min(int(D * scale_factor * rng.uniform(1.0 - noise, 1.0 + noise)), MAX_D))

    # Feasibility guarantee: D must be > min signup time
    min_signup = min(lib['signup'] for lib in new_libraries)
    if new_D <= min_signup:
        new_D = min(max(new_D, int(min_signup * 1.5)), MAX_D)

    return new_B, new_L, new_D, new_scores, new_libraries


def compute_instance_stats(B, L, D, scores, libraries, seed_name="", scale=0.0, tightness_param=0.0):
    """Compute statistics about an instance for paper reporting."""
    signups = [lib['signup'] for lib in libraries]
    ship_rates = [lib['ship_rate'] for lib in libraries]
    scores_arr = np.array(scores, dtype=np.float64)
    signups_arr = np.array(signups, dtype=np.float64)
    ship_rates_arr = np.array(ship_rates, dtype=np.float64)

    total_signup = sum(signups)
    actual_tightness = D / total_signup if total_signup > 0 else float('inf')

    # Score statistics
    score_mean = scores_arr.mean()
    score_std = scores_arr.std()
    score_cv = score_std / score_mean if score_mean > 0 else 0.0

    # Sampled Jaccard overlap (50 random pairs)
    rng_stats = np.random.default_rng(0)
    jaccard_samples = []
    if L >= 2:
        n_pairs = min(50, L * (L - 1) // 2)
        for _ in range(n_pairs):
            i, j = rng_stats.choice(L, size=2, replace=False)
            set_i = set(libraries[i]['books'])
            set_j = set(libraries[j]['books'])
            union = len(set_i | set_j)
            if union > 0:
                jaccard_samples.append(len(set_i & set_j) / union)
            else:
                jaccard_samples.append(0.0)

    jaccard_mean = np.mean(jaccard_samples) if jaccard_samples else 0.0

    # Book coverage: fraction of books that appear in at least one library
    all_books = set()
    for lib in libraries:
        all_books.update(lib['books'])
    book_coverage = len(all_books) / B if B > 0 else 0.0

    # Library size stats
    lib_sizes = [lib['n_books'] for lib in libraries]

    return {
        'seed': seed_name,
        'scale': scale,
        'tightness_param': tightness_param,
        'B': B,
        'L': L,
        'D': D,
        'total_signup': total_signup,
        'actual_tightness': round(actual_tightness, 4),
        'signup_mean': round(signups_arr.mean(), 2),
        'signup_std': round(signups_arr.std(), 2),
        'signup_min': int(signups_arr.min()),
        'signup_max': int(signups_arr.max()),
        'ship_rate_mean': round(ship_rates_arr.mean(), 2),
        'ship_rate_std': round(ship_rates_arr.std(), 2),
        'score_mean': round(score_mean, 2),
        'score_std': round(score_std, 2),
        'score_cv': round(score_cv, 4),
        'jaccard_overlap_mean': round(jaccard_mean, 4),
        'book_coverage': round(book_coverage, 4),
        'lib_size_mean': round(np.mean(lib_sizes), 2),
        'lib_size_std': round(np.std(lib_sizes), 2),
    }


def instance_filename(seed_letter, scale, tightness, replicate=None):
    """Generate filename: {seed_letter}_{scale}x_{tightness}t[_r{N}].txt"""
    scale_int = str(round(scale * 100)).zfill(3)
    tightness_int = str(round(tightness * 100)).zfill(3)
    base = f"{seed_letter}_{scale_int}x_{tightness_int}t"
    if replicate is not None:
        base += f"_r{replicate}"
    return base + ".txt"


def generate_batch(seed_dir, out_dir, random_seed=42, scales=None,
                    tightness_levels=None, replicates=1, include_seeds=False, noise=None):
    """Generate instances using a systematic grid of seeds x scales x tightness x replicates."""
    if scales is None:
        scales = [0.25, 0.5, 0.75, 1.0, 1.5]
    if tightness_levels is None:
        tightness_levels = [0.1, 0.25, 0.5, 0.75, 1.0]
    if noise is None:
        noise = 0.2

    # Check for filename collisions (custom grids with close decimal values)
    test_names = set()
    for s in scales:
        for t in tightness_levels:
            for r in range(1, replicates + 1):
                name = instance_filename('x', s, t, replicate=r if replicates > 1 else None)
                if name in test_names:
                    print(f"Error: scale={s} and tightness={t} produce a duplicate filename '{name}'.")
                    print("Use values that differ by at least 0.01 to avoid collisions.")
                    sys.exit(1)
                test_names.add(name)

    # Auto-discover seed files
    seed_files = sorted(glob.glob(os.path.join(seed_dir, '*.txt')))
    if not seed_files:
        print(f"Error: No seed files found in {seed_dir}")
        sys.exit(1)

    n_generated = len(seed_files) * len(scales) * len(tightness_levels) * replicates
    print(f"Found {len(seed_files)} seed files: {[os.path.basename(f) for f in seed_files]}")
    print(f"Grid: {len(seed_files)} seeds x {len(scales)} scales x {len(tightness_levels)} tightness"
          f" x {replicates} replicates = {n_generated} instances")
    print(f"Scales: {scales}")
    print(f"Tightness: {tightness_levels}")
    print(f"Noise: {noise}, Random seed: {random_seed}")
    print()

    os.makedirs(out_dir, exist_ok=True)

    all_stats = []
    count = 0

    # Optionally copy original seed files into output
    if include_seeds:
        print("--- Including original seed instances ---")
        for seed_file in seed_files:
            dest = os.path.join(out_dir, os.path.basename(seed_file))
            shutil.copy2(seed_file, dest)
            print(f"  Copied {os.path.basename(seed_file)}")

            # Also compute stats for seed instances
            B, L, D, scores, libraries = read_instance(seed_file)
            seed_letter = detect_seed_letter(seed_file)
            stats = compute_instance_stats(
                B, L, D, scores, libraries,
                seed_name=f"{seed_letter}_orig", scale=1.0, tightness_param=0.0
            )
            all_stats.append(stats)
        print()

    for seed_file in seed_files:
        seed_letter = detect_seed_letter(seed_file)
        if seed_letter is None:
            print(f"Warning: Could not detect seed letter from {seed_file}, skipping")
            continue

        print(f"--- Seed: {os.path.basename(seed_file)} (letter={seed_letter}) ---")
        B, L, D, scores, libraries = read_instance(seed_file)

        for scale in scales:
            for tightness in tightness_levels:
                for rep in range(1, replicates + 1):
                    count += 1
                    # Deterministic RNG: hashlib instead of hash() for cross-platform reproducibility
                    key = f"{random_seed}_{seed_letter}_{scale}_{tightness}_{rep}"
                    instance_seed = int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**31)
                    rng = np.random.default_rng(instance_seed)

                    t0 = time.time()
                    nB, nL, nD, nScores, nLibs = generate_new_instance(
                        B, L, D, scores, libraries, scale, noise, rng,
                        tightness=tightness, seed_letter=seed_letter
                    )

                    validate_instance(nB, nL, nD, nScores, nLibs)

                    rep_tag = rep if replicates > 1 else None
                    fname = instance_filename(seed_letter, scale, tightness, replicate=rep_tag)
                    fpath = os.path.join(out_dir, fname)
                    write_instance(fpath, nB, nL, nD, nScores, nLibs)
                    elapsed = time.time() - t0

                    stats = compute_instance_stats(
                        nB, nL, nD, nScores, nLibs,
                        seed_name=seed_letter, scale=scale, tightness_param=tightness
                    )
                    if replicates > 1:
                        stats['replicate'] = rep
                    all_stats.append(stats)

                    print(f"  [{count}/{n_generated}] {fname}  "
                          f"(B={nB}, L={nL}, D={nD}, tightness={stats['actual_tightness']}) "
                          f"— {elapsed:.2f}s")

    # Write summary CSV
    csv_path = os.path.join(out_dir, 'summary.csv')
    if all_stats:
        fieldnames = list(all_stats[0].keys())
        # Ensure 'replicate' column is present if any row has it
        if any('replicate' in s for s in all_stats):
            if 'replicate' not in fieldnames:
                fieldnames.append('replicate')
            for s in all_stats:
                s.setdefault('replicate', 1)
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_stats)
        print(f"\nSummary CSV written to {csv_path} ({len(all_stats)} instances)")

    return all_stats


def main():
    parser = argparse.ArgumentParser(description="Generate new HashCode 2020 Book Scanning instances from seeds.")
    parser.add_argument('seed_file', type=str, nargs='?', default=None,
                        help="Path to the seed instance file (not needed with --batch)")
    parser.add_argument('--count', type=int, default=1, help="Number of instances to generate")
    parser.add_argument('--scale', type=float, default=1.0, help="Scale factor for size")
    parser.add_argument('--noise', type=float, default=0.2, help="Noise factor (e.g. 0.2 for 20%% perturbation)")
    parser.add_argument('--out_dir', type=str, default='generated_instances', help="Output directory")
    parser.add_argument('--seed', type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument('--tightness', type=float, default=None,
                        help="Tightness parameter (0.1=hard, 1.0=easy). Controls D relative to total signup.")
    parser.add_argument('--batch', action='store_true',
                        help="Batch mode: generate instances from a grid of seeds x scales x tightness x replicates")
    parser.add_argument('--seed_dir', type=str, default='seed',
                        help="Directory containing seed files (used with --batch)")
    parser.add_argument('--scales', type=float, nargs='+', default=None,
                        help="Scale factors for batch mode (default: 0.25 0.5 0.75 1.0 1.5)")
    parser.add_argument('--tightness_levels', type=float, nargs='+', default=None,
                        help="Tightness levels for batch mode (default: 0.1 0.25 0.5 0.75 1.0)")
    parser.add_argument('--replicates', type=int, default=1,
                        help="Number of replicates per (seed, scale, tightness) combination (default: 1)")
    parser.add_argument('--include_seeds', action='store_true',
                        help="Copy original seed instances into the output directory and include in summary")

    args = parser.parse_args()

    if args.batch:
        generate_batch(
            args.seed_dir, args.out_dir,
            random_seed=args.seed if args.seed is not None else 42,
            scales=args.scales,
            tightness_levels=args.tightness_levels,
            replicates=args.replicates,
            include_seeds=args.include_seeds,
            noise=args.noise,
        )
        return

    # Single-instance mode (backward compatible)
    if args.seed_file is None:
        parser.error("seed_file is required when not using --batch mode")

    rng = np.random.default_rng(args.seed)
    seed_letter = detect_seed_letter(args.seed_file)

    print(f"Reading seed: {args.seed_file}")
    B, L, D, scores, libraries = read_instance(args.seed_file)
    print(f"Original: {B} books, {L} libraries, {D} days")

    base_name = os.path.basename(args.seed_file).split('.')[0]

    os.makedirs(args.out_dir, exist_ok=True)

    for i in range(args.count):
        t0 = time.time()
        nB, nL, nD, nScores, nLibs = generate_new_instance(
            B, L, D, scores, libraries, args.scale, args.noise, rng,
            tightness=args.tightness, seed_letter=seed_letter
        )

        out_name = f"{base_name}_s{args.scale}_n{args.noise}_{i+1}.txt"
        out_path = os.path.join(args.out_dir, out_name)

        write_instance(out_path, nB, nL, nD, nScores, nLibs)
        elapsed = time.time() - t0
        print(f"[{i+1}/{args.count}] {out_path}  ({nB} books, {nL} libs, {nD} days) — {elapsed:.1f}s")

        # Print stats if requested via tightness
        stats = compute_instance_stats(nB, nL, nD, nScores, nLibs,
                                       seed_name=base_name, scale=args.scale,
                                       tightness_param=args.tightness or 0.0)
        print(f"  Stats: tightness={stats['actual_tightness']}, "
              f"score_cv={stats['score_cv']}, "
              f"jaccard={stats['jaccard_overlap_mean']}, "
              f"coverage={stats['book_coverage']}")


if __name__ == "__main__":
    main()

import sys
import argparse
import os
import csv
import glob
import hashlib
import shutil
from itertools import combinations
from collections import defaultdict
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
TIGHTNESS_REL_TOL = 0.15
TIGHTNESS_ABS_TOL = 0.02
MAX_GENERATION_ATTEMPTS = 8
OVERLAP_PRESETS = {
    'default': {'popularity_gamma': 1.0, 'core_fraction': 0.0, 'core_weight': 0.0},
    'hard': {'popularity_gamma': 1.8, 'core_fraction': 0.05, 'core_weight': 0.35},
}
STRUCTURE_PRESETS = {
    'default': {
        'group_fraction': 0.0,
        'core_ratio': 0.0,
        'core_take_ratio': 0.0,
        'pool_ratio': 0.0,
        'signup_compression': 0.0,
        'rate_compression': 0.0,
    },
    'competitive': {
        'group_fraction': 0.08,
        'core_ratio': 0.35,
        'core_take_ratio': 0.55,
        'pool_ratio': 1.75,
        'signup_compression': 0.45,
        'rate_compression': 0.35,
    },
}
SOURCE_REGIME_PRESETS = {
    'default': ('default', 'default'),
    'hard': ('hard', 'competitive'),
}


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
    """Extract the leading alphabetic seed code from a filename."""
    basename = os.path.basename(filepath).lower()
    if basename and basename[0].isalpha():
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


def deterministic_instance_seed(random_seed, source_name, scale, tightness, attempt,
                                overlap_mode='default', popularity_gamma=1.0,
                                core_fraction=0.0, core_weight=0.0,
                                structure_mode='default', group_fraction=0.0,
                                structure_core_ratio=0.0, structure_core_take_ratio=0.0,
                                structure_pool_ratio=0.0, signup_compression=0.0,
                                rate_compression=0.0):
    """Cross-platform deterministic seed for one generation attempt."""
    key = (
        f"{random_seed}_{source_name}_{scale}_{tightness}_{attempt}_"
        f"{overlap_mode}_{popularity_gamma}_{core_fraction}_{core_weight}_"
        f"{structure_mode}_{group_fraction}_{structure_core_ratio}_"
        f"{structure_core_take_ratio}_{structure_pool_ratio}_"
        f"{signup_compression}_{rate_compression}"
    )
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**31)


def tightness_tolerance(target):
    return max(TIGHTNESS_ABS_TOL, abs(target) * TIGHTNESS_REL_TOL)


def resolve_overlap_config(overlap_mode, popularity_gamma=None,
                           core_fraction=None, core_weight=None):
    """Resolve overlap-generation parameters from a preset plus optional overrides."""
    if overlap_mode not in OVERLAP_PRESETS:
        raise ValueError(f"Unknown overlap_mode='{overlap_mode}'. Expected one of {sorted(OVERLAP_PRESETS)}")

    preset = OVERLAP_PRESETS[overlap_mode]
    gamma = preset['popularity_gamma'] if popularity_gamma is None else popularity_gamma
    core_frac = preset['core_fraction'] if core_fraction is None else core_fraction
    core_mass = preset['core_weight'] if core_weight is None else core_weight

    if gamma <= 0:
        raise ValueError("popularity_gamma must be > 0")
    if not (0.0 <= core_frac < 1.0):
        raise ValueError("core_fraction must be in [0, 1)")
    if not (0.0 <= core_mass < 1.0):
        raise ValueError("core_weight must be in [0, 1)")

    return {
        'overlap_mode': overlap_mode,
        'popularity_gamma': float(gamma),
        'core_fraction': float(core_frac),
        'core_weight': float(core_mass),
    }


def resolve_structure_config(structure_mode, group_fraction=None, core_ratio=None,
                             core_take_ratio=None, pool_ratio=None,
                             signup_compression=None, rate_compression=None):
    """Resolve structure-generation parameters from a preset plus optional overrides."""
    if structure_mode not in STRUCTURE_PRESETS:
        raise ValueError(f"Unknown structure_mode='{structure_mode}'. Expected one of {sorted(STRUCTURE_PRESETS)}")

    preset = STRUCTURE_PRESETS[structure_mode]
    config = {
        'structure_mode': structure_mode,
        'group_fraction': preset['group_fraction'] if group_fraction is None else group_fraction,
        'core_ratio': preset['core_ratio'] if core_ratio is None else core_ratio,
        'core_take_ratio': preset['core_take_ratio'] if core_take_ratio is None else core_take_ratio,
        'pool_ratio': preset['pool_ratio'] if pool_ratio is None else pool_ratio,
        'signup_compression': preset['signup_compression'] if signup_compression is None else signup_compression,
        'rate_compression': preset['rate_compression'] if rate_compression is None else rate_compression,
    }

    bounded_keys = [
        'group_fraction', 'core_ratio', 'core_take_ratio',
        'pool_ratio', 'signup_compression', 'rate_compression',
    ]
    for key in bounded_keys:
        if config[key] < 0.0:
            raise ValueError(f"{key} must be >= 0")
    if config['group_fraction'] >= 1.0:
        raise ValueError("group_fraction must be in [0, 1)")
    if config['core_ratio'] >= 1.0:
        raise ValueError("core_ratio must be in [0, 1)")
    if config['core_take_ratio'] >= 1.0:
        raise ValueError("core_take_ratio must be in [0, 1)")
    if config['signup_compression'] >= 1.0:
        raise ValueError("signup_compression must be in [0, 1)")
    if config['rate_compression'] >= 1.0:
        raise ValueError("rate_compression must be in [0, 1)")

    return {key: float(value) if key != 'structure_mode' else value for key, value in config.items()}


def parse_source_regimes(spec):
    """Parse per-source regimes like 'b:hard,c:default,be:hard'."""
    if not spec:
        return {}

    mapping = {}
    for token in str(spec).split(','):
        token = token.strip()
        if not token:
            continue
        if ':' not in token:
            raise ValueError(
                f"Invalid source regime token '{token}'. Expected format like 'b:hard'."
            )
        source, regime = token.split(':', 1)
        source = source.strip()
        regime = regime.strip()
        if regime not in SOURCE_REGIME_PRESETS:
            raise ValueError(
                f"Unknown source regime '{regime}' for source '{source}'. "
                f"Expected one of {sorted(SOURCE_REGIME_PRESETS)}."
            )
        mapping[source] = regime
    return mapping


def resolve_source_generation_config(source_name, source_regimes, default_overlap_config,
                                     default_structure_config):
    """Resolve the overlap/structure config for one source."""
    regime = source_regimes.get(source_name)
    if regime is None:
        return default_overlap_config, default_structure_config, 'global'

    overlap_mode, structure_mode = SOURCE_REGIME_PRESETS[regime]
    return (
        resolve_overlap_config(overlap_mode),
        resolve_structure_config(structure_mode),
        regime,
    )


def blend_toward_median(value, median_value, strength):
    """Move a value toward the median while keeping positive integer bounds."""
    if strength <= 0.0:
        return int(value)
    blended = (1.0 - strength) * value + strength * median_value
    return max(1, int(round(blended)))


def build_competitive_group_templates(new_B, new_L, avg_target_n_books, rng,
                                      popularity_weights, score_weights, structure_config):
    """Create group-level shared cores/pools that induce hard library competition."""
    if structure_config['structure_mode'] == 'default' or new_L <= 1:
        return None, None

    n_groups = int(round(new_L * structure_config['group_fraction']))
    n_groups = max(3, min(24, n_groups))
    n_groups = min(n_groups, new_L)

    group_ids = np.arange(new_L) % n_groups
    rng.shuffle(group_ids)

    mean_size = max(8, int(round(avg_target_n_books)))
    core_size = max(4, min(new_B, int(round(mean_size * structure_config['core_ratio']))))
    pool_size = max(core_size + 8, min(new_B, int(round(mean_size * structure_config['pool_ratio']))))

    combined_weights = popularity_weights * score_weights
    combined_weights = combined_weights / combined_weights.sum()

    group_templates = []
    for _ in range(n_groups):
        pool_books = rng.choice(new_B, size=pool_size, replace=False, p=combined_weights)
        core_books = pool_books[:core_size]
        extra_pool = pool_books[core_size:]
        group_templates.append({
            'core': core_books.tolist(),
            'pool': extra_pool.tolist(),
        })

    return group_ids.tolist(), group_templates


def assign_books_competitive(target_n_books, rng, popularity_weights, group_template,
                             structure_config):
    """Assign books using a shared group core plus unique tail books."""
    target_n_books = min(target_n_books, len(popularity_weights))
    if target_n_books <= 0:
        return []

    selected = []
    selected_set = set()

    core_books = group_template['core']
    pool_books = group_template['pool']

    desired_core = min(len(core_books), target_n_books, int(round(target_n_books * structure_config['core_take_ratio'])))
    if desired_core > 0:
        core_weights = np.array([popularity_weights[bid] for bid in core_books], dtype=np.float64)
        core_weights = core_weights / core_weights.sum()
        chosen_core = rng.choice(np.array(core_books, dtype=np.int64), size=desired_core, replace=False, p=core_weights)
        for bid in chosen_core.tolist():
            if bid not in selected_set:
                selected.append(bid)
                selected_set.add(bid)

    remaining = target_n_books - len(selected)
    if remaining > 0 and pool_books:
        pool_candidates = [bid for bid in pool_books if bid not in selected_set]
        if pool_candidates:
            take = min(remaining, len(pool_candidates))
            pool_weights = np.array([popularity_weights[bid] for bid in pool_candidates], dtype=np.float64)
            pool_weights = pool_weights / pool_weights.sum()
            chosen_pool = rng.choice(np.array(pool_candidates, dtype=np.int64), size=take, replace=False, p=pool_weights)
            for bid in chosen_pool.tolist():
                if bid not in selected_set:
                    selected.append(bid)
                    selected_set.add(bid)

    remaining = target_n_books - len(selected)
    if remaining > 0:
        all_books = np.arange(len(popularity_weights))
        retries = 0
        while remaining > 0 and retries < 4:
            sample_size = min(len(popularity_weights), max(remaining * 3, remaining + 32))
            candidates = rng.choice(all_books, size=sample_size, replace=False, p=popularity_weights)
            added = 0
            for bid in candidates.tolist():
                if bid not in selected_set:
                    selected.append(bid)
                    selected_set.add(bid)
                    added += 1
                    remaining -= 1
                    if remaining == 0:
                        break
            if added == 0:
                retries += 1

    return selected[:target_n_books]


def generate_popularity_weights(new_B, rng, popularity_gamma=1.0,
                                core_fraction=0.0, core_weight=0.0):
    """Generate popularity weights for books using an exponential distribution.

    A small number of books will be very popular (high weight) while most
    books have moderate popularity, creating natural overlap when libraries
    sample from this distribution.
    """
    raw_weights = rng.exponential(scale=1.0, size=new_B)
    if popularity_gamma != 1.0:
        raw_weights = np.power(raw_weights, popularity_gamma)

    weights = raw_weights / raw_weights.sum()
    if core_fraction > 0.0 and core_weight > 0.0:
        core_size = min(new_B, max(1, int(round(new_B * core_fraction))))
        core_indices = rng.choice(new_B, size=core_size, replace=False)
        weights *= (1.0 - core_weight)
        core_weights = raw_weights[core_indices]
        core_weights = core_weights / core_weights.sum()
        weights[core_indices] += core_weight * core_weights

    return weights


def generate_new_instance(B, L, D, scores, libraries, scale_factor, noise, rng,
                          tightness=None, seed_letter=None, overlap_config=None,
                          structure_config=None):
    """Generates a new instance by scaling and perturbing the seed instance."""
    # Apply noise to B and L dimensions, clamped to Hash Code bounds
    new_B = max(1, min(int(B * scale_factor * rng.uniform(1.0 - noise, 1.0 + noise)), MAX_B))
    new_L = max(1, min(int(L * scale_factor * rng.uniform(1.0 - noise, 1.0 + noise)), MAX_L))

    # Generate scores using improved method
    if seed_letter is None:
        seed_letter = 'c'  # default to heterogeneous behavior
    new_scores = generate_scores(scores, new_B, seed_letter, noise, rng)

    if overlap_config is None:
        overlap_config = resolve_overlap_config('default')
    if structure_config is None:
        structure_config = resolve_structure_config('default')

    # Generate popularity weights for book assignment
    popularity_weights = generate_popularity_weights(
        new_B,
        rng,
        popularity_gamma=overlap_config['popularity_gamma'],
        core_fraction=overlap_config['core_fraction'],
        core_weight=overlap_config['core_weight'],
    )
    score_weights = np.asarray(new_scores, dtype=np.float64) + 1.0
    score_weights = score_weights / score_weights.sum()

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

    avg_target_n_books = np.mean([lib['n_books'] for lib in libraries]) * scale_factor if libraries else 1.0
    group_ids, group_templates = build_competitive_group_templates(
        new_B,
        new_L,
        avg_target_n_books,
        rng,
        popularity_weights,
        score_weights,
        structure_config,
    )

    signup_values = np.array([lib['signup'] for lib in libraries], dtype=np.float64)
    rate_values = np.array([lib['ship_rate'] for lib in libraries], dtype=np.float64)
    signup_median = float(np.median(signup_values)) if len(signup_values) > 0 else 1.0
    rate_median = float(np.median(rate_values)) if len(rate_values) > 0 else 1.0

    new_libraries = []
    total_signup = 0
    for i in range(new_L):
        orig_lib = libraries[lib_indices[i]]

        # Perturb signup and ship rate, clamped to Hash Code bounds
        raw_signup = max(1, min(int(orig_lib['signup'] * rng.uniform(1.0 - noise, 1.0 + noise)), MAX_T))
        raw_ship_rate = max(1, min(int(orig_lib['ship_rate'] * rng.uniform(1.0 - noise, 1.0 + noise)), MAX_M))
        new_signup = blend_toward_median(raw_signup, signup_median, structure_config['signup_compression'])
        new_ship_rate = blend_toward_median(raw_ship_rate, rate_median, structure_config['rate_compression'])
        new_signup = max(1, min(new_signup, MAX_T))
        new_ship_rate = max(1, min(new_ship_rate, MAX_M))

        # Determine target number of books (clamped to new_B and MAX_N)
        target_n_books = max(1, min(
            int(orig_lib['n_books'] * scale_factor * rng.uniform(1.0 - noise, 1.0 + noise)),
            new_B, MAX_N
        ))

        # Popularity-weighted or competitive group assignment
        if group_templates is None:
            new_books = assign_books_with_overlap(new_B, target_n_books, rng, popularity_weights)
        else:
            template = group_templates[group_ids[i]]
            new_books = assign_books_competitive(
                target_n_books,
                rng,
                popularity_weights,
                template,
                structure_config,
            )

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

    # Compute D based on tightness or scale from seed, clamped to MAX_D.
    # When a requested tightness would violate D <= MAX_D, we deterministically
    # clip to the maximum feasible actual-tightness target for this candidate.
    requested_tightness = None
    effective_tightness = None
    max_feasible_tightness = None
    tightness_clipped = False
    if tightness is not None:
        requested_tightness = float(tightness)
        max_feasible_tightness = min(1.0, MAX_D / total_signup) if total_signup > 0 else 1.0
        effective_tightness = min(requested_tightness, max_feasible_tightness)
        tightness_clipped = effective_tightness < requested_tightness
        new_D = max(1, min(int(total_signup * effective_tightness), MAX_D))
    else:
        new_D = max(1, min(int(D * scale_factor * rng.uniform(1.0 - noise, 1.0 + noise)), MAX_D))

    # Feasibility guarantee: D must be > min signup time so at least one
    # library can ship at least one book (a library with T_j signup days
    # can only start shipping on day T_j, so D must be >= T_j + 1)
    min_signup = min(lib['signup'] for lib in new_libraries)
    if new_D <= min_signup:
        new_D = min(min_signup + max(1, min_signup // 2), MAX_D)

    metadata = {
        'requested_tightness': requested_tightness,
        'effective_target_tightness': effective_tightness,
        'max_feasible_tightness': max_feasible_tightness,
        'tightness_clipped': tightness_clipped,
        'overlap_mode': overlap_config['overlap_mode'],
        'popularity_gamma': overlap_config['popularity_gamma'],
        'core_fraction': overlap_config['core_fraction'],
        'core_weight': overlap_config['core_weight'],
        'structure_mode': structure_config['structure_mode'],
        'group_fraction': structure_config['group_fraction'],
        'structure_core_ratio': structure_config['core_ratio'],
        'structure_core_take_ratio': structure_config['core_take_ratio'],
        'structure_pool_ratio': structure_config['pool_ratio'],
        'signup_compression': structure_config['signup_compression'],
        'rate_compression': structure_config['rate_compression'],
    }

    return new_B, new_L, new_D, new_scores, new_libraries, metadata


def compute_instance_stats(
    B, L, D, scores, libraries, seed_name="", scale=0.0, tightness_param=0.0,
    effective_target_tightness=0.0, max_feasible_tightness=0.0,
    tightness_clipped=False, generation_attempt=0, overlap_mode='default',
    popularity_gamma=1.0, core_fraction=0.0, core_weight=0.0,
    structure_mode='default', group_fraction=0.0, structure_core_ratio=0.0,
    structure_core_take_ratio=0.0, structure_pool_ratio=0.0,
    signup_compression=0.0, rate_compression=0.0,
):
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
    score_variance = float(scores_arr.var())
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

    # Book duplication rate: fraction of books appearing in 2+ libraries
    book_freq = defaultdict(int)
    for lib in libraries:
        for book_id in lib['books']:
            book_freq[book_id] += 1
    duplicated_books = sum(1 for freq in book_freq.values() if freq >= 2)
    book_duplication_rate = duplicated_books / B if B > 0 else 0.0

    return {
        'seed': seed_name,
        'scale': scale,
        'tightness_param': tightness_param,
        'effective_target_tightness': round(effective_target_tightness, 4),
        'max_feasible_tightness': round(max_feasible_tightness, 4),
        'tightness_clipped': int(bool(tightness_clipped)),
        'generation_attempt': generation_attempt,
        'overlap_mode': overlap_mode,
        'popularity_gamma': round(popularity_gamma, 4),
        'core_fraction': round(core_fraction, 4),
        'core_weight': round(core_weight, 4),
        'structure_mode': structure_mode,
        'group_fraction': round(group_fraction, 4),
        'structure_core_ratio': round(structure_core_ratio, 4),
        'structure_core_take_ratio': round(structure_core_take_ratio, 4),
        'structure_pool_ratio': round(structure_pool_ratio, 4),
        'signup_compression': round(signup_compression, 4),
        'rate_compression': round(rate_compression, 4),
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
        'score_variance': round(score_variance, 2),
        'score_cv': round(score_cv, 4),
        'jaccard_overlap_mean': round(jaccard_mean, 4),
        'book_coverage': round(book_coverage, 4),
        'book_duplication_rate': round(book_duplication_rate, 4),
        'lib_size_mean': round(np.mean(lib_sizes), 2),
        'lib_size_std': round(np.std(lib_sizes), 2),
    }


def instance_filename(seed_letter, scale, tightness):
    """Generate filename: {seed_letter}_{scale}x_{tightness}t.txt"""
    scale_int = str(round(scale * 100)).zfill(3)
    tightness_int = str(round(tightness * 100)).zfill(3)
    return f"{seed_letter}_{scale_int}x_{tightness_int}t.txt"


def generate_batch(seed_dir, out_dir, random_seed=42, scales=None,
                    tightness_levels=None, include_seeds=False, noise=None,
                    crossbreed=False, overlap_mode='default',
                    popularity_gamma=None, core_fraction=None, core_weight=None,
                    structure_mode='default', group_fraction=None,
                    structure_core_ratio=None, structure_core_take_ratio=None,
                    structure_pool_ratio=None, signup_compression=None,
                    rate_compression=None, source_regimes=None):
    """Generate instances using a systematic grid of seeds x scales x tightness."""
    if scales is None:
        if crossbreed:
            scales = [0.25, 0.5, 1.0]          # 15 sources x 3 x 3 = 135
        else:
            scales = [0.25, 0.5, 0.75, 1.0]    # 5 seeds x 4 x 5 = 100
    if tightness_levels is None:
        if crossbreed:
            tightness_levels = [0.1, 0.5, 1.0]
        else:
            tightness_levels = [0.1, 0.25, 0.5, 0.75, 1.0]
    if noise is None:
        noise = 0.2
    overlap_config = resolve_overlap_config(
        overlap_mode,
        popularity_gamma=popularity_gamma,
        core_fraction=core_fraction,
        core_weight=core_weight,
    )
    structure_config = resolve_structure_config(
        structure_mode,
        group_fraction=group_fraction,
        core_ratio=structure_core_ratio,
        core_take_ratio=structure_core_take_ratio,
        pool_ratio=structure_pool_ratio,
        signup_compression=signup_compression,
        rate_compression=rate_compression,
    )
    source_regime_map = parse_source_regimes(source_regimes)

    # Check for filename collisions (custom grids with close decimal values)
    test_names = set()
    for s in scales:
        for t in tightness_levels:
            name = instance_filename('x', s, t)
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

    # Pre-read all seeds and build unified source list
    seed_cache = {}
    for sf in seed_files:
        letter = detect_seed_letter(sf)
        if letter is None:
            print(f"Warning: Could not detect seed letter from {sf}, skipping")
            continue
        seed_cache[sf] = (letter, *read_instance(sf))

    # Sources: each entry is (name, B, L, D, scores, libraries, score_seed_letter)
    sources = []
    for sf in seed_files:
        if sf not in seed_cache:
            continue
        letter, B, L, D, scores, libs = seed_cache[sf]
        sources.append((letter, B, L, D, scores, libs, letter))

    if crossbreed and len(seed_cache) >= 2:
        for sf_a, sf_b in combinations(seed_cache.keys(), 2):
            la, Ba, _, _, sa, _ = seed_cache[sf_a]
            lb, _, Lb, Db, _, libs_b = seed_cache[sf_b]
            sources.append((f"{la}{lb}", Ba, Lb, Db, sa, libs_b, la))

    n_generated = len(sources) * len(scales) * len(tightness_levels)
    n_single = len([s for s in sources if len(s[0]) == 1])
    n_cross = len(sources) - n_single
    print(f"Found {len(seed_cache)} seed files: {[os.path.basename(f) for f in seed_files if f in seed_cache]}")
    if crossbreed and n_cross > 0:
        print(f"Sources: {n_single} seeds + {n_cross} cross-bred pairs = {len(sources)} sources")
    print(f"Grid: {len(sources)} sources x {len(scales)} scales x {len(tightness_levels)} tightness"
          f" = {n_generated} instances")
    print(f"Scales: {scales}")
    print(f"Tightness: {tightness_levels}")
    print(f"Noise: {noise}, Random seed: {random_seed}")
    print(
        "Overlap: "
        f"mode={overlap_config['overlap_mode']}, "
        f"gamma={overlap_config['popularity_gamma']}, "
        f"core_fraction={overlap_config['core_fraction']}, "
        f"core_weight={overlap_config['core_weight']}"
    )
    print(
        "Structure: "
        f"mode={structure_config['structure_mode']}, "
        f"group_fraction={structure_config['group_fraction']}, "
        f"core_ratio={structure_config['core_ratio']}, "
        f"core_take_ratio={structure_config['core_take_ratio']}, "
        f"pool_ratio={structure_config['pool_ratio']}, "
        f"signup_compression={structure_config['signup_compression']}, "
        f"rate_compression={structure_config['rate_compression']}"
    )
    if source_regime_map:
        print(f"Source-specific regimes: {source_regime_map}")
    print()

    os.makedirs(out_dir, exist_ok=True)

    all_stats = []
    count = 0

    # Optionally copy original seed files into output
    if include_seeds:
        print("--- Including original seed instances ---")
        for sf in seed_files:
            if sf not in seed_cache:
                continue
            dest = os.path.join(out_dir, os.path.basename(sf))
            shutil.copy2(sf, dest)
            letter, B, L, D, scores, libs = seed_cache[sf]
            print(f"  Copied {os.path.basename(sf)}")
            stats = compute_instance_stats(
                B, L, D, scores, libs,
                seed_name=f"{letter}_orig", scale=1.0, tightness_param=0.0,
                effective_target_tightness=0.0, max_feasible_tightness=0.0,
                tightness_clipped=False, generation_attempt=0,
                overlap_mode='seed',
                popularity_gamma=0.0,
                core_fraction=0.0,
                core_weight=0.0,
                structure_mode='seed',
                group_fraction=0.0,
                structure_core_ratio=0.0,
                structure_core_take_ratio=0.0,
                structure_pool_ratio=0.0,
                signup_compression=0.0,
                rate_compression=0.0,
            )
            all_stats.append(stats)
        print()

    for src_name, B, L, D, scores, libraries, score_letter in sources:
        is_cross = len(src_name) > 1
        src_overlap_config, src_structure_config, src_regime = resolve_source_generation_config(
            src_name,
            source_regime_map,
            overlap_config,
            structure_config,
        )
        if is_cross:
            print(
                f"--- Cross-breed: {src_name[0]} (scores) x {src_name[1]} (structure) "
                f"[regime={src_regime}] ---"
            )
        else:
            print(f"--- Seed: {src_name} [regime={src_regime}] ---")

        for scale in scales:
            for tightness in tightness_levels:
                count += 1
                t0 = time.time()
                best_candidate = None
                best_gap = None
                selected_attempt = 0

                for attempt in range(MAX_GENERATION_ATTEMPTS):
                    instance_seed = deterministic_instance_seed(
                        random_seed, src_name, scale, tightness, attempt,
                        overlap_mode=src_overlap_config['overlap_mode'],
                        popularity_gamma=src_overlap_config['popularity_gamma'],
                        core_fraction=src_overlap_config['core_fraction'],
                        core_weight=src_overlap_config['core_weight'],
                        structure_mode=src_structure_config['structure_mode'],
                        group_fraction=src_structure_config['group_fraction'],
                        structure_core_ratio=src_structure_config['core_ratio'],
                        structure_core_take_ratio=src_structure_config['core_take_ratio'],
                        structure_pool_ratio=src_structure_config['pool_ratio'],
                        signup_compression=src_structure_config['signup_compression'],
                        rate_compression=src_structure_config['rate_compression'],
                    )
                    rng = np.random.default_rng(instance_seed)
                    candidate = generate_new_instance(
                        B, L, D, scores, libraries, scale, noise, rng,
                        tightness=tightness, seed_letter=score_letter,
                        overlap_config=src_overlap_config,
                        structure_config=src_structure_config,
                    )
                    nB, nL, nD, nScores, nLibs, meta = candidate
                    target = meta['effective_target_tightness']
                    actual = (nD / sum(lib['signup'] for lib in nLibs)) if nLibs else 0.0
                    gap = abs(actual - target) if target is not None else 0.0

                    if best_candidate is None or gap < best_gap:
                        best_candidate = candidate
                        best_gap = gap
                        selected_attempt = attempt

                    if target is None or gap <= tightness_tolerance(target):
                        break

                nB, nL, nD, nScores, nLibs, meta = best_candidate

                validate_instance(nB, nL, nD, nScores, nLibs)

                fname = instance_filename(src_name, scale, tightness)
                fpath = os.path.join(out_dir, fname)
                write_instance(fpath, nB, nL, nD, nScores, nLibs)
                elapsed = time.time() - t0

                stats = compute_instance_stats(
                    nB, nL, nD, nScores, nLibs,
                    seed_name=src_name,
                    scale=scale,
                    tightness_param=tightness,
                    effective_target_tightness=meta['effective_target_tightness'] or 0.0,
                    max_feasible_tightness=meta['max_feasible_tightness'] or 0.0,
                    tightness_clipped=meta['tightness_clipped'],
                    generation_attempt=selected_attempt,
                    overlap_mode=meta['overlap_mode'],
                    popularity_gamma=meta['popularity_gamma'],
                    core_fraction=meta['core_fraction'],
                    core_weight=meta['core_weight'],
                    structure_mode=meta['structure_mode'],
                    group_fraction=meta['group_fraction'],
                    structure_core_ratio=meta['structure_core_ratio'],
                    structure_core_take_ratio=meta['structure_core_take_ratio'],
                    structure_pool_ratio=meta['structure_pool_ratio'],
                    signup_compression=meta['signup_compression'],
                    rate_compression=meta['rate_compression'],
                )
                all_stats.append(stats)

                print(f"  [{count}/{n_generated}] {fname}  "
                      f"(B={nB}, L={nL}, D={nD}, actual={stats['actual_tightness']}, "
                      f"target={stats['effective_target_tightness']}) "
                      f"— {elapsed:.2f}s")

    # Write summary CSV
    csv_path = os.path.join(out_dir, 'summary.csv')
    if all_stats:
        fieldnames = list(all_stats[0].keys())
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
                        help="Batch mode: generate instances from a grid of seeds x scales x tightness")
    parser.add_argument('--seed_dir', type=str, default='seed',
                        help="Directory containing seed files (used with --batch)")
    parser.add_argument('--scales', type=float, nargs='+', default=None,
                        help="Scale factors for batch mode (default: 0.25 0.5 0.75 1.0; with --crossbreed: 0.25 0.5 1.0)")
    parser.add_argument('--tightness_levels', type=float, nargs='+', default=None,
                        help="Tightness levels for batch mode (default: 0.1 0.25 0.5 0.75 1.0; with --crossbreed: 0.1 0.5 1.0)")
    parser.add_argument('--include_seeds', action='store_true',
                        help="Copy original seed instances into the output directory and include in summary")
    parser.add_argument('--crossbreed', action='store_true',
                        help="Generate hybrid instances by cross-breeding pairs of seeds "
                             "(scores from one seed, library structure from another)")
    parser.add_argument('--overlap_mode', choices=sorted(OVERLAP_PRESETS), default='default',
                        help="Overlap preset for book assignment. 'hard' increases duplication/conflict.")
    parser.add_argument('--popularity_gamma', type=float, default=None,
                        help="Optional override for popularity concentration (>1 gives more overlap).")
    parser.add_argument('--core_fraction', type=float, default=None,
                        help="Optional override for the fraction of books treated as a shared hot core.")
    parser.add_argument('--core_weight', type=float, default=None,
                        help="Optional override for the probability mass assigned to the hot-core books.")
    parser.add_argument('--structure_mode', choices=sorted(STRUCTURE_PRESETS), default='default',
                        help="Library structure preset. 'competitive' makes libraries more confusable for greedy.")
    parser.add_argument('--group_fraction', type=float, default=None,
                        help="Optional override for the fraction of library groups in competitive mode.")
    parser.add_argument('--structure_core_ratio', type=float, default=None,
                        help="Optional override for the shared core size as a fraction of average library size.")
    parser.add_argument('--structure_core_take_ratio', type=float, default=None,
                        help="Optional override for how much of each library comes from the shared core.")
    parser.add_argument('--structure_pool_ratio', type=float, default=None,
                        help="Optional override for group pool size relative to average library size.")
    parser.add_argument('--signup_compression', type=float, default=None,
                        help="Optional override for compression of signup times toward the median.")
    parser.add_argument('--rate_compression', type=float, default=None,
                        help="Optional override for compression of ship rates toward the median.")
    parser.add_argument('--source_regimes', type=str, default=None,
                        help="Optional per-source regimes like 'b:default,c:default,e:hard,be:hard'. "
                             "Regimes: default=(default/default), hard=(hard/competitive).")

    args = parser.parse_args()

    if args.batch:
        generate_batch(
            args.seed_dir, args.out_dir,
            random_seed=args.seed if args.seed is not None else 42,
            scales=args.scales,
            tightness_levels=args.tightness_levels,
            include_seeds=args.include_seeds,
            noise=args.noise,
            crossbreed=args.crossbreed,
            overlap_mode=args.overlap_mode,
            popularity_gamma=args.popularity_gamma,
            core_fraction=args.core_fraction,
            core_weight=args.core_weight,
            structure_mode=args.structure_mode,
            group_fraction=args.group_fraction,
            structure_core_ratio=args.structure_core_ratio,
            structure_core_take_ratio=args.structure_core_take_ratio,
            structure_pool_ratio=args.structure_pool_ratio,
            signup_compression=args.signup_compression,
            rate_compression=args.rate_compression,
            source_regimes=args.source_regimes,
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
    overlap_config = resolve_overlap_config(
        args.overlap_mode,
        popularity_gamma=args.popularity_gamma,
        core_fraction=args.core_fraction,
        core_weight=args.core_weight,
    )
    structure_config = resolve_structure_config(
        args.structure_mode,
        group_fraction=args.group_fraction,
        core_ratio=args.structure_core_ratio,
        core_take_ratio=args.structure_core_take_ratio,
        pool_ratio=args.structure_pool_ratio,
        signup_compression=args.signup_compression,
        rate_compression=args.rate_compression,
    )
    print(
        "Overlap config: "
        f"mode={overlap_config['overlap_mode']}, "
        f"gamma={overlap_config['popularity_gamma']}, "
        f"core_fraction={overlap_config['core_fraction']}, "
        f"core_weight={overlap_config['core_weight']}"
    )
    print(
        "Structure config: "
        f"mode={structure_config['structure_mode']}, "
        f"group_fraction={structure_config['group_fraction']}, "
        f"core_ratio={structure_config['core_ratio']}, "
        f"core_take_ratio={structure_config['core_take_ratio']}, "
        f"pool_ratio={structure_config['pool_ratio']}, "
        f"signup_compression={structure_config['signup_compression']}, "
        f"rate_compression={structure_config['rate_compression']}"
    )

    base_name = os.path.basename(args.seed_file).split('.')[0]

    os.makedirs(args.out_dir, exist_ok=True)

    for i in range(args.count):
        t0 = time.time()
        nB, nL, nD, nScores, nLibs, meta = generate_new_instance(
            B, L, D, scores, libraries, args.scale, args.noise, rng,
            tightness=args.tightness, seed_letter=seed_letter,
            overlap_config=overlap_config,
            structure_config=structure_config,
        )

        out_name = f"{base_name}_s{args.scale}_n{args.noise}_{i+1}.txt"
        out_path = os.path.join(args.out_dir, out_name)

        write_instance(out_path, nB, nL, nD, nScores, nLibs)
        elapsed = time.time() - t0
        print(f"[{i+1}/{args.count}] {out_path}  ({nB} books, {nL} libs, {nD} days) — {elapsed:.1f}s")

        # Print stats if requested via tightness
        stats = compute_instance_stats(nB, nL, nD, nScores, nLibs,
                                       seed_name=base_name, scale=args.scale,
                                       tightness_param=args.tightness or 0.0,
                                       effective_target_tightness=meta['effective_target_tightness'] or 0.0,
                                       max_feasible_tightness=meta['max_feasible_tightness'] or 0.0,
                                       tightness_clipped=meta['tightness_clipped'],
                                       generation_attempt=0,
                                       overlap_mode=meta['overlap_mode'],
                                       popularity_gamma=meta['popularity_gamma'],
                                       core_fraction=meta['core_fraction'],
                                       core_weight=meta['core_weight'],
                                       structure_mode=meta['structure_mode'],
                                       group_fraction=meta['group_fraction'],
                                       structure_core_ratio=meta['structure_core_ratio'],
                                       structure_core_take_ratio=meta['structure_core_take_ratio'],
                                       structure_pool_ratio=meta['structure_pool_ratio'],
                                       signup_compression=meta['signup_compression'],
                                       rate_compression=meta['rate_compression'])
        print(f"  Stats: tightness={stats['actual_tightness']}, "
              f"score_cv={stats['score_cv']}, "
              f"jaccard={stats['jaccard_overlap_mean']}, "
              f"coverage={stats['book_coverage']}")


if __name__ == "__main__":
    main()

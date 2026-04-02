import sys
import argparse
import os
import csv
import glob
import hashlib
import json
import shutil
from itertools import combinations
from collections import defaultdict
import numpy as np
import time

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

DEFAULT_SEED_PROFILE = {
    'structure_base': 0.58,
    'structure_min': 0.35,
    'structure_max': 0.68,
    'size_penalty_scale': 0.60,
    'size_penalty_cap': 0.18,
    'score_blend_base': 0.08,
    'score_blend_slope': 0.18,
    'score_blend_cap': 0.25,
    'seed_freq_blend': 0.40,
    'base_weight_power': 1.0,
    'seed_freq_power': 1.0,
    'signup_noise_mult': 1.0,
    'ship_rate_noise_mult': 1.0,
    'book_count_noise_mult': 1.0,
    'anchor_ratio_scale': 1.0,
    'shared_top_pool_frac': 0.0,
    'decoy_lib_frac': 0.0,
    'decoy_book_frac': 0.0,
    'inversion_pool_frac': 0.0,
    'slow_lib_frac': 0.0,
    'slow_book_frac': 0.0,
    'gem_pool_frac': 0.0,
    'gem_lib_frac': 0.0,
    'gem_book_frac': 0.0,
    'size_downscale_mult': 1.0,
    'size_upscale_mult': 1.0,
}

SEED_PROFILES = {
    # b is very stable under density-based greedy, so we reduce template
    # anchoring and inject a few more overlapping / delayed high-value books.
    'b': {
        'structure_base': 0.42,
        'structure_min': 0.18,
        'structure_max': 0.52,
        'size_penalty_scale': 0.22,
        'size_penalty_cap': 0.08,
        'score_blend_base': 0.04,
        'score_blend_slope': 0.08,
        'score_blend_cap': 0.12,
        'seed_freq_blend': 0.25,
        'base_weight_power': 1.25,
        'signup_noise_mult': 1.55,
        'ship_rate_noise_mult': 1.70,
        'book_count_noise_mult': 1.30,
        'anchor_ratio_scale': 0.82,
        'shared_top_pool_frac': 0.050,
        'decoy_lib_frac': 0.22,
        'decoy_book_frac': 0.20,
        'inversion_pool_frac': 0.035,
        'slow_lib_frac': 0.20,
        'slow_book_frac': 0.18,
        'gem_pool_frac': 0.020,
        'gem_lib_frac': 0.10,
        'gem_book_frac': 0.22,
        'size_downscale_mult': 0.30,
        'size_upscale_mult': 1.18,
    },
    'c': {
        'structure_base': 0.48,
        'structure_min': 0.22,
        'structure_max': 0.60,
        'size_penalty_scale': 0.38,
        'size_penalty_cap': 0.12,
        'score_blend_base': 0.06,
        'score_blend_slope': 0.12,
        'score_blend_cap': 0.18,
        'seed_freq_blend': 0.32,
        'base_weight_power': 1.10,
        'signup_noise_mult': 1.25,
        'ship_rate_noise_mult': 1.30,
        'book_count_noise_mult': 1.18,
        'anchor_ratio_scale': 0.88,
        'shared_top_pool_frac': 0.035,
        'decoy_lib_frac': 0.18,
        'decoy_book_frac': 0.14,
        'inversion_pool_frac': 0.025,
        'slow_lib_frac': 0.16,
        'slow_book_frac': 0.12,
        'gem_pool_frac': 0.014,
        'gem_lib_frac': 0.06,
        'gem_book_frac': 0.16,
        'size_downscale_mult': 0.45,
        'size_upscale_mult': 1.12,
    },
    # d needs lighter structural anchoring and stronger parameter variation,
    # otherwise the generated instances remain too close to greedy-optimal.
    'd': {
        'structure_base': 0.30,
        'structure_min': 0.12,
        'structure_max': 0.40,
        'size_penalty_scale': 0.30,
        'size_penalty_cap': 0.08,
        'score_blend_base': 0.03,
        'score_blend_slope': 0.05,
        'score_blend_cap': 0.10,
        'seed_freq_blend': 0.25,
        'base_weight_power': 1.35,
        'seed_freq_power': 1.20,
        'signup_noise_mult': 1.70,
        'ship_rate_noise_mult': 2.10,
        'book_count_noise_mult': 1.35,
        'anchor_ratio_scale': 0.80,
        'shared_top_pool_frac': 0.045,
        'decoy_lib_frac': 0.20,
        'decoy_book_frac': 0.18,
        'inversion_pool_frac': 0.030,
        'slow_lib_frac': 0.18,
        'slow_book_frac': 0.14,
        'gem_pool_frac': 0.018,
        'gem_lib_frac': 0.08,
        'gem_book_frac': 0.18,
        'size_downscale_mult': 0.40,
        'size_upscale_mult': 1.15,
    },
}


def get_seed_profile(seed_letter):
    profile = dict(DEFAULT_SEED_PROFILE)
    profile.update(SEED_PROFILES.get(seed_letter, {}))
    return profile


def safe_corrcoef(x, y):
    """Correlation helper that returns 0.0 for degenerate inputs."""
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if len(x_arr) < 2 or len(y_arr) < 2:
        return 0.0
    if np.allclose(x_arr.std(), 0.0) or np.allclose(y_arr.std(), 0.0):
        return 0.0
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


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


def generate_scores(orig_scores, new_B, noise, rng):
    """Generate book scores from the empirical seed distribution.

    We use one unified model for every seed:
    bootstrap from the original scores, apply multiplicative noise, then add a
    small centered perturbation. This keeps the seed's score scale and
    distribution shape while still introducing mild diversity, including for
    low-variance seeds such as b/d.
    """
    orig = np.array(orig_scores, dtype=np.float64)
    mean_score = orig.mean()
    orig_std = orig.std()

    resampled = rng.choice(orig, size=new_B, replace=True)
    noise_vals = rng.uniform(1.0 - noise, 1.0 + noise, size=new_B)
    new_scores = resampled * noise_vals

    if orig_std < 1e-6:
        additive_scale = max(1.0, mean_score * noise * 0.25)
    else:
        additive_scale = max(1.0, orig_std * noise * 0.15)
    new_scores += rng.normal(loc=0.0, scale=additive_scale, size=new_B)

    # Recenter so the generated mean stays faithful to the source seed.
    new_scores += (mean_score - new_scores.mean())

    # Clamp to [0, MAX_SCORE] per Hash Code spec and convert to int
    new_scores = np.clip(np.round(new_scores), 0, MAX_SCORE).astype(np.int64)
    return new_scores.tolist()


def build_seed_book_projection(orig_scores, new_scores, libraries):
    """Project seed-book identity onto generated books via score-rank alignment.

    This lets each generated library preserve a core of the source library's
    content instead of rebuilding everything from a global pool.
    """
    orig_scores_arr = np.asarray(orig_scores, dtype=np.float64)
    new_scores_arr = np.asarray(new_scores, dtype=np.float64)

    orig_order = np.argsort(-orig_scores_arr, kind='stable')
    new_order = np.argsort(-new_scores_arr, kind='stable')
    overlap = min(len(orig_order), len(new_order))

    orig_to_new = {int(orig_order[i]): int(new_order[i]) for i in range(overlap)}

    seed_book_freq = np.zeros(len(orig_scores_arr), dtype=np.float64)
    for lib in libraries:
        for book_id in lib['books']:
            if 0 <= book_id < len(seed_book_freq):
                seed_book_freq[book_id] += 1.0

    projected_freq = np.ones(len(new_scores_arr), dtype=np.float64)
    if overlap > 0:
        projected_freq[new_order[:overlap]] = seed_book_freq[orig_order[:overlap]] + 1.0

    if len(new_order) > overlap and len(orig_order) > 0:
        seed_freq_sorted = seed_book_freq[orig_order] + 1.0
        extra_positions = np.linspace(
            0, len(seed_freq_sorted) - 1,
            num=len(new_order) - overlap,
            dtype=int,
        )
        projected_freq[new_order[overlap:]] = seed_freq_sorted[extra_positions]

    return orig_to_new, projected_freq


def assign_books_with_overlap(new_B, target_n_books, rng, popularity_weights, anchor_books=None):
    """Assign books using weighted sampling plus an optional structural core.

    Popular books appear in many libraries naturally, creating realistic
    overlap patterns without expensive pairwise computation.
    """
    target_n_books = min(target_n_books, new_B)
    chosen = []

    if anchor_books:
        anchor_candidates = list(dict.fromkeys(int(book_id) for book_id in anchor_books))
        anchor_take = min(len(anchor_candidates), target_n_books)
        if anchor_take > 0:
            if len(anchor_candidates) == anchor_take:
                anchor_selected = anchor_candidates
            else:
                anchor_weights = popularity_weights[anchor_candidates]
                anchor_weights = anchor_weights / anchor_weights.sum()
                anchor_selected = rng.choice(
                    anchor_candidates,
                    size=anchor_take,
                    replace=False,
                    p=anchor_weights,
                ).tolist()
            chosen.extend(anchor_selected)

    remaining = target_n_books - len(chosen)
    if remaining <= 0:
        return chosen

    weights = popularity_weights.copy()
    if chosen:
        weights[chosen] = 0.0
        weight_sum = weights.sum()
        if weight_sum <= 0:
            remaining_pool = [idx for idx in range(new_B) if idx not in set(chosen)]
            extra = rng.choice(remaining_pool, size=remaining, replace=False).tolist()
            chosen.extend(extra)
            return chosen
        weights /= weight_sum

    extra = rng.choice(new_B, size=remaining, replace=False, p=weights)
    chosen.extend(extra.tolist())
    return chosen


def pick_weighted_books(pool, take, rng, chosen_set=None):
    """Sample a small subset while avoiding duplicates.

    We intentionally keep this lightweight because it runs once per library.
    The pools are already score-ranked, so uniform sampling within them keeps
    the anti-greedy signal without adding much generation cost.
    """
    if take <= 0 or not pool:
        return []
    filtered = [int(book_id) for book_id in pool if chosen_set is None or int(book_id) not in chosen_set]
    if not filtered:
        return []
    take = min(take, len(filtered))
    if len(filtered) == take:
        return filtered
    return rng.choice(filtered, size=take, replace=False).tolist()


def deterministic_instance_seed(random_seed, source_name, scale, tightness, attempt):
    """Cross-platform deterministic seed for one generation attempt."""
    key = f"{random_seed}_{source_name}_{scale}_{tightness}_{attempt}"
    return int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**31)


def tightness_tolerance(target):
    return max(TIGHTNESS_ABS_TOL, abs(target) * TIGHTNESS_REL_TOL)


def generate_popularity_weights(scores, rng, projected_seed_freq=None, profile=None):
    """Generate book-selection weights with mild score/popularity correlation.

    The base exponential component keeps overlap natural, while a light score
    prior helps preserve the intuition that better books tend to appear in more
    strategically relevant libraries. When available, we also blend in the
    source seed's empirical book-frequency prior.
    """
    profile = profile or DEFAULT_SEED_PROFILE
    scores_arr = np.asarray(scores, dtype=np.float64)
    base_weights = rng.exponential(scale=1.0, size=len(scores_arr))
    if profile['base_weight_power'] != 1.0:
        base_weights = np.power(base_weights, profile['base_weight_power'])
    base_weights = base_weights / base_weights.sum()

    score_weights = scores_arr + 1.0
    score_weights = score_weights / score_weights.sum()
    score_cv = scores_arr.std() / scores_arr.mean() if scores_arr.mean() > 0 else 0.0
    score_blend = min(
        profile['score_blend_cap'],
        profile['score_blend_base'] + profile['score_blend_slope'] * score_cv,
    )

    if projected_seed_freq is None:
        blended = (1.0 - score_blend) * base_weights + score_blend * score_weights
        return blended / blended.sum()

    seed_freq_weights = np.asarray(projected_seed_freq, dtype=np.float64)
    if profile['seed_freq_power'] != 1.0:
        seed_freq_weights = np.power(seed_freq_weights, profile['seed_freq_power'])
    seed_freq_weights = seed_freq_weights / seed_freq_weights.sum()
    seed_freq_blend = profile['seed_freq_blend']
    base_blend = 1.0 - score_blend - seed_freq_blend
    blended = (
        base_blend * base_weights
        + score_blend * score_weights
        + seed_freq_blend * seed_freq_weights
    )
    return blended / blended.sum()


def write_generation_config(out_dir, payload):
    """Persist reproducibility metadata for paper/reporting."""
    config_path = os.path.join(out_dir, 'generation_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def generate_new_instance(B, L, D, scores, libraries, scale_factor, noise, rng,
                          tightness=None, seed_letter=None, structure_seed_letter=None):
    """Generates a new instance by scaling and perturbing the seed instance."""
    if seed_letter is None:
        seed_letter = 'c'  # default to heterogeneous behavior
    if structure_seed_letter is None:
        structure_seed_letter = seed_letter
    score_profile = get_seed_profile(seed_letter)
    structure_profile = get_seed_profile(structure_seed_letter)

    # Apply a mild upward size bias for seed families that otherwise generate
    # very low-score, greedy-friendly instances.
    down_mult = min(score_profile['size_downscale_mult'], structure_profile['size_downscale_mult'])
    up_mult = max(score_profile['size_upscale_mult'], structure_profile['size_upscale_mult'])
    size_low = max(0.05, 1.0 - noise * down_mult)
    size_high = 1.0 + noise * up_mult

    # Apply noise to B and L dimensions, clamped to Hash Code bounds
    new_B = max(1, min(int(B * scale_factor * rng.uniform(size_low, size_high)), MAX_B))
    new_L = max(1, min(int(L * scale_factor * rng.uniform(size_low, size_high)), MAX_L))

    # Generate scores using improved method
    new_scores = generate_scores(scores, new_B, noise, rng)

    orig_to_new, projected_seed_freq = build_seed_book_projection(scores, new_scores, libraries)

    # Generate popularity weights for book assignment
    popularity_weights = generate_popularity_weights(new_scores, rng, projected_seed_freq, score_profile)
    score_weights = np.asarray(new_scores, dtype=np.float64) + 1.0

    ranked_books = np.argsort(-score_weights)
    shared_take = min(len(ranked_books), int(round(new_B * structure_profile['shared_top_pool_frac'])))
    shared_pool = ranked_books[:shared_take].tolist()
    rest_ranked = ranked_books[shared_take:].tolist()
    inversion_take = min(len(rest_ranked), int(round(new_B * structure_profile['inversion_pool_frac'])))
    inversion_pool = rest_ranked[:inversion_take]
    rest_ranked = rest_ranked[inversion_take:]
    gem_take = min(len(rest_ranked), int(round(new_B * structure_profile['gem_pool_frac'])))
    gem_pool = rest_ranked[:gem_take]

    orig_signups = np.asarray([lib['signup'] for lib in libraries], dtype=np.float64)
    orig_sizes = np.asarray([lib['n_books'] for lib in libraries], dtype=np.float64)
    orig_density = np.asarray([
        (lib['ship_rate'] * min(lib['n_books'], max(1, D - lib['signup']))) / max(1, lib['signup'])
        for lib in libraries
    ], dtype=np.float64)

    dense_threshold = None
    if structure_profile['decoy_lib_frac'] > 0.0 and len(orig_density) > 0:
        dense_threshold = float(np.quantile(orig_density, max(0.0, 1.0 - structure_profile['decoy_lib_frac'])))
    slow_threshold = None
    if structure_profile['slow_lib_frac'] > 0.0 and len(orig_signups) > 0:
        slow_threshold = float(np.quantile(orig_signups, max(0.0, 1.0 - structure_profile['slow_lib_frac'])))
    small_threshold = None
    if structure_profile['gem_lib_frac'] > 0.0 and len(orig_sizes) > 0:
        small_threshold = float(np.quantile(orig_sizes, min(1.0, structure_profile['gem_lib_frac'])))

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
    template_core_fractions = []
    for i in range(new_L):
        orig_lib = libraries[lib_indices[i]]

        # Perturb signup and ship rate, clamped to Hash Code bounds
        signup_noise = noise * structure_profile['signup_noise_mult']
        ship_rate_noise = noise * structure_profile['ship_rate_noise_mult']
        book_count_noise = noise * structure_profile['book_count_noise_mult']

        new_signup = max(
            1,
            min(int(orig_lib['signup'] * rng.uniform(1.0 - signup_noise, 1.0 + signup_noise)), MAX_T),
        )
        new_ship_rate = max(
            1,
            min(int(orig_lib['ship_rate'] * rng.uniform(1.0 - ship_rate_noise, 1.0 + ship_rate_noise)), MAX_M),
        )

        # Determine target number of books (clamped to new_B and MAX_N)
        target_n_books = max(1, min(
            int(orig_lib['n_books'] * scale_factor * rng.uniform(1.0 - book_count_noise, 1.0 + book_count_noise)),
            new_B, MAX_N
        ))

        mapped_template_books = [
            orig_to_new[book_id]
            for book_id in orig_lib['books']
            if book_id in orig_to_new
        ]
        # Preserve local library identity, but lower the core for very large
        # libraries so the generator remains diverse and solver-informative.
        lib_size_share = orig_lib['n_books'] / max(1, B)
        size_penalty = min(
            structure_profile['size_penalty_cap'],
            lib_size_share * structure_profile['size_penalty_scale'],
        )
        structure_ratio = min(
            structure_profile['structure_max'],
            max(structure_profile['structure_min'], structure_profile['structure_base'] - 0.5 * noise - size_penalty),
        )
        structure_ratio *= structure_profile['anchor_ratio_scale']
        structure_ratio = min(structure_profile['structure_max'], max(structure_profile['structure_min'], structure_ratio))
        anchor_target = min(
            len(mapped_template_books),
            max(1, int(round(target_n_books * structure_ratio))),
        ) if mapped_template_books else 0
        anchor_books = None
        if anchor_target > 0:
            if len(mapped_template_books) <= anchor_target:
                anchor_books = mapped_template_books
            else:
                template_weights = popularity_weights[mapped_template_books]
                template_weights = template_weights / template_weights.sum()
                anchor_books = rng.choice(
                    mapped_template_books,
                    size=anchor_target,
                    replace=False,
                    p=template_weights,
                ).tolist()

        role_books = []
        chosen_role = set()
        orig_density_val = (
            (orig_lib['ship_rate'] * min(orig_lib['n_books'], max(1, D - orig_lib['signup'])))
            / max(1, orig_lib['signup'])
        )
        is_decoy = (
            dense_threshold is not None
            and orig_density_val >= dense_threshold
            and structure_profile['decoy_book_frac'] > 0.0
        )
        is_slow = (
            slow_threshold is not None
            and orig_lib['signup'] >= slow_threshold
            and structure_profile['slow_book_frac'] > 0.0
        )
        is_gem = (
            small_threshold is not None
            and orig_lib['n_books'] <= small_threshold
            and structure_profile['gem_book_frac'] > 0.0
        )

        if is_decoy and shared_pool:
            role_take = max(1, int(round(target_n_books * structure_profile['decoy_book_frac'])))
            picked = pick_weighted_books(shared_pool, role_take, rng, chosen_set=chosen_role)
            role_books.extend(picked)
            chosen_role.update(picked)

        if is_slow and inversion_pool:
            role_take = max(1, int(round(target_n_books * structure_profile['slow_book_frac'])))
            picked = pick_weighted_books(inversion_pool, role_take, rng, chosen_set=chosen_role)
            role_books.extend(picked)
            chosen_role.update(picked)

        if is_gem and gem_pool:
            role_take = max(1, int(round(target_n_books * structure_profile['gem_book_frac'])))
            picked = pick_weighted_books(gem_pool, role_take, rng, chosen_set=chosen_role)
            role_books.extend(picked)
            chosen_role.update(picked)

        if role_books:
            merged_anchor = []
            seen_anchor = set()
            for collection in (role_books, anchor_books or []):
                for book_id in collection:
                    book_id = int(book_id)
                    if book_id not in seen_anchor:
                        merged_anchor.append(book_id)
                        seen_anchor.add(book_id)
            anchor_books = merged_anchor

        new_books = assign_books_with_overlap(
            new_B, target_n_books, rng, popularity_weights, anchor_books=anchor_books
        )
        template_core_fractions.append(anchor_target / max(1, len(new_books)))

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
        'template_core_ratio_mean': float(np.mean(template_core_fractions)) if template_core_fractions else 0.0,
        'profile_name': seed_letter,
        'structure_profile_name': structure_seed_letter,
    }

    return new_B, new_L, new_D, new_scores, new_libraries, metadata


def compute_instance_stats(
    B, L, D, scores, libraries, seed_name="", scale=0.0, tightness_param=0.0,
    effective_target_tightness=0.0, max_feasible_tightness=0.0,
    tightness_clipped=False, generation_attempt=0, tightness_gap=0.0,
    tightness_warning=False, template_core_ratio_mean=0.0, reference_stats=None,
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
    score_sum = float(scores_arr.sum())
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
    lib_sizes_arr = np.array(lib_sizes, dtype=np.float64)

    # Book duplication rate: fraction of books appearing in 2+ libraries
    book_freq = defaultdict(int)
    for lib in libraries:
        for book_id in lib['books']:
            book_freq[book_id] += 1
    duplicated_books = sum(1 for freq in book_freq.values() if freq >= 2)
    book_duplication_rate = duplicated_books / B if B > 0 else 0.0

    book_freq_arr = np.zeros(B, dtype=np.float64)
    for book_id, freq in book_freq.items():
        if 0 <= book_id < B:
            book_freq_arr[book_id] = freq

    signup_ship_rate_corr = safe_corrcoef(signups_arr, ship_rates_arr)
    signup_lib_size_corr = safe_corrcoef(signups_arr, lib_sizes_arr)
    lib_size_ship_rate_corr = safe_corrcoef(lib_sizes_arr, ship_rates_arr)
    score_book_freq_corr = safe_corrcoef(scores_arr, book_freq_arr)

    min_signup = int(signups_arr.min()) if len(signups_arr) > 0 else 0
    feasibility_ratio = (D / min_signup) if min_signup > 0 else 0.0

    total_effective_books = 0.0
    total_library_books = 0.0
    for lib in libraries:
        potential = max(0, D - lib['signup']) * lib['ship_rate']
        total_effective_books += min(lib['n_books'], potential)
        total_library_books += lib['n_books']
    effective_books_ratio = (total_effective_books / total_library_books) if total_library_books > 0 else 0.0

    stats = {
        'seed': seed_name,
        'scale': scale,
        'tightness_param': tightness_param,
        'effective_target_tightness': round(effective_target_tightness, 4),
        'max_feasible_tightness': round(max_feasible_tightness, 4),
        'tightness_clipped': int(bool(tightness_clipped)),
        'generation_attempt': generation_attempt,
        'tightness_gap': round(tightness_gap, 4),
        'tightness_warning': int(bool(tightness_warning)),
        'template_core_ratio_mean': round(template_core_ratio_mean, 4),
        'B': B,
        'L': L,
        'D': D,
        'total_signup': total_signup,
        'actual_tightness': round(actual_tightness, 4),
        'feasibility_ratio': round(feasibility_ratio, 4),
        'effective_books_ratio': round(effective_books_ratio, 4),
        'signup_mean': round(signups_arr.mean(), 2),
        'signup_std': round(signups_arr.std(), 2),
        'signup_min': min_signup,
        'signup_max': int(signups_arr.max()),
        'ship_rate_mean': round(ship_rates_arr.mean(), 2),
        'ship_rate_std': round(ship_rates_arr.std(), 2),
        'score_sum': int(round(score_sum)),
        'score_mean': round(score_mean, 2),
        'score_std': round(score_std, 2),
        'score_variance': round(score_variance, 2),
        'score_cv': round(score_cv, 4),
        'score_book_freq_corr': round(score_book_freq_corr, 4),
        'signup_ship_rate_corr': round(signup_ship_rate_corr, 4),
        'signup_lib_size_corr': round(signup_lib_size_corr, 4),
        'lib_size_ship_rate_corr': round(lib_size_ship_rate_corr, 4),
        'jaccard_overlap_mean': round(jaccard_mean, 4),
        'book_coverage': round(book_coverage, 4),
        'book_duplication_rate': round(book_duplication_rate, 4),
        'lib_size_mean': round(np.mean(lib_sizes), 2),
        'lib_size_std': round(np.std(lib_sizes), 2),
    }

    if reference_stats is not None:
        stats.update({
            'score_mean_delta_pct': round(
                ((stats['score_mean'] - reference_stats['score_mean']) / reference_stats['score_mean'] * 100.0)
                if reference_stats['score_mean'] else 0.0,
                4,
            ),
            'score_cv_delta': round(stats['score_cv'] - reference_stats['score_cv'], 4),
            'signup_mean_delta_pct': round(
                ((stats['signup_mean'] - reference_stats['signup_mean']) / reference_stats['signup_mean'] * 100.0)
                if reference_stats['signup_mean'] else 0.0,
                4,
            ),
            'ship_rate_mean_delta_pct': round(
                ((stats['ship_rate_mean'] - reference_stats['ship_rate_mean']) / reference_stats['ship_rate_mean'] * 100.0)
                if reference_stats['ship_rate_mean'] else 0.0,
                4,
            ),
            'lib_size_mean_delta_pct': round(
                ((stats['lib_size_mean'] - reference_stats['lib_size_mean']) / reference_stats['lib_size_mean'] * 100.0)
                if reference_stats['lib_size_mean'] else 0.0,
                4,
            ),
            'jaccard_delta': round(stats['jaccard_overlap_mean'] - reference_stats['jaccard_overlap_mean'], 4),
            'score_book_freq_corr_delta': round(
                stats['score_book_freq_corr'] - reference_stats['score_book_freq_corr'], 4
            ),
            'signup_ship_rate_corr_delta': round(
                stats['signup_ship_rate_corr'] - reference_stats['signup_ship_rate_corr'], 4
            ),
            'signup_lib_size_corr_delta': round(
                stats['signup_lib_size_corr'] - reference_stats['signup_lib_size_corr'], 4
            ),
            'lib_size_ship_rate_corr_delta': round(
                stats['lib_size_ship_rate_corr'] - reference_stats['lib_size_ship_rate_corr'], 4
            ),
        })

    return stats


def instance_filename(seed_letter, scale, tightness):
    """Generate filename: {seed_letter}_{scale}x_{tightness}t.txt"""
    scale_int = str(round(scale * 100)).zfill(3)
    tightness_int = str(round(tightness * 100)).zfill(3)
    return f"{seed_letter}_{scale_int}x_{tightness_int}t.txt"


def generate_batch(seed_dir, out_dir, random_seed=42, scales=None,
                    tightness_levels=None, include_seeds=False, noise=None,
                    crossbreed=False):
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
            la, Ba, _La, _Da, sa, _libs_a = seed_cache[sf_a]
            lb, _Bb, Lb, Db, _sb, libs_b = seed_cache[sf_b]
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
    print()

    os.makedirs(out_dir, exist_ok=True)
    write_generation_config(out_dir, {
        'mode': 'batch',
        'random_seed': random_seed,
        'noise': noise,
        'scales': scales,
        'tightness_levels': tightness_levels,
        'crossbreed': bool(crossbreed),
        'include_seeds': bool(include_seeds),
        'seed_dir': seed_dir,
        'seed_files': [os.path.basename(f) for f in seed_files if f in seed_cache],
        'sources': [src[0] for src in sources],
    })

    all_stats = []
    count = 0
    source_reference_stats = {
        src_name: compute_instance_stats(B, L, D, scores, libraries, seed_name=src_name)
        for src_name, B, L, D, scores, libraries, _score_letter in sources
    }

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
                reference_stats=source_reference_stats[letter],
            )
            all_stats.append(stats)
        print()

    for src_name, B, L, D, scores, libraries, score_letter in sources:
        is_cross = len(src_name) > 1
        if is_cross:
            print(f"--- Cross-breed: {src_name[0]} (scores) x {src_name[1]} (structure) ---")
        else:
            print(f"--- Seed: {src_name} ---")

        for scale in scales:
            for tightness in tightness_levels:
                count += 1
                t0 = time.time()
                best_candidate = None
                best_gap = None
                selected_attempt = 0

                for attempt in range(MAX_GENERATION_ATTEMPTS):
                    instance_seed = deterministic_instance_seed(random_seed, src_name, scale, tightness, attempt)
                    rng = np.random.default_rng(instance_seed)
                    candidate = generate_new_instance(
                        B, L, D, scores, libraries, scale, noise, rng,
                        tightness=tightness, seed_letter=score_letter,
                        structure_seed_letter=src_name[-1],
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
                    tightness_gap=best_gap or 0.0,
                    tightness_warning=(
                        meta['effective_target_tightness'] is not None
                        and best_gap is not None
                        and best_gap > tightness_tolerance(meta['effective_target_tightness'])
                    ),
                    template_core_ratio_mean=meta.get('template_core_ratio_mean', 0.0),
                    reference_stats=source_reference_stats[src_name],
                )
                all_stats.append(stats)

                if stats['tightness_warning']:
                    print(
                        f"    WARNING tightness gap={stats['tightness_gap']} "
                        f"(target={stats['effective_target_tightness']}, actual={stats['actual_tightness']})"
                    )

                print(f"  [{count}/{n_generated}] {fname}  "
                      f"(B={nB}, L={nL}, D={nD}, actual={stats['actual_tightness']}, "
                      f"target={stats['effective_target_tightness']}) "
                      f"- {elapsed:.2f}s")

    # Write summary CSV
    csv_path = os.path.join(out_dir, 'summary.csv')
    if all_stats:
        fieldnames = []
        for row in all_stats:
            for key in row.keys():
                if key not in fieldnames:
                    fieldnames.append(key)
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_stats)
        print(f"\nSummary CSV written to {csv_path} ({len(all_stats)} instances)")
        print(
            "Diversity check: "
            f"score_cv=[{min(s['score_cv'] for s in all_stats):.4f}, {max(s['score_cv'] for s in all_stats):.4f}], "
            f"actual_tightness=[{min(s['actual_tightness'] for s in all_stats):.4f}, {max(s['actual_tightness'] for s in all_stats):.4f}], "
            f"tightness_warnings={sum(s['tightness_warning'] for s in all_stats)}"
        )

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
    write_generation_config(args.out_dir, {
        'mode': 'single',
        'random_seed': args.seed,
        'noise': args.noise,
        'scale': args.scale,
        'tightness': args.tightness,
        'seed_file': args.seed_file,
        'count': args.count,
    })
    reference_stats = compute_instance_stats(B, L, D, scores, libraries, seed_name=base_name)

    for i in range(args.count):
        t0 = time.time()
        nB, nL, nD, nScores, nLibs, meta = generate_new_instance(
            B, L, D, scores, libraries, args.scale, args.noise, rng,
            tightness=args.tightness, seed_letter=seed_letter,
            structure_seed_letter=seed_letter,
        )

        out_name = f"{base_name}_s{args.scale}_n{args.noise}_{i+1}.txt"
        out_path = os.path.join(args.out_dir, out_name)

        write_instance(out_path, nB, nL, nD, nScores, nLibs)
        elapsed = time.time() - t0
        print(f"[{i+1}/{args.count}] {out_path}  ({nB} books, {nL} libs, {nD} days) - {elapsed:.1f}s")

        # Print stats if requested via tightness
        stats = compute_instance_stats(nB, nL, nD, nScores, nLibs,
                                       seed_name=base_name, scale=args.scale,
                                       tightness_param=args.tightness or 0.0,
                                       effective_target_tightness=meta['effective_target_tightness'] or 0.0,
                                       max_feasible_tightness=meta['max_feasible_tightness'] or 0.0,
                                       tightness_clipped=meta['tightness_clipped'],
                                       generation_attempt=0,
                                       template_core_ratio_mean=meta.get('template_core_ratio_mean', 0.0),
                                       reference_stats=reference_stats)
        print(f"  Stats: tightness={stats['actual_tightness']}, "
              f"score_cv={stats['score_cv']}, "
              f"jaccard={stats['jaccard_overlap_mean']}, "
              f"coverage={stats['book_coverage']}, "
              f"template_core={stats['template_core_ratio_mean']}")


if __name__ == "__main__":
    main()

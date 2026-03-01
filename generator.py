import sys
import random
import argparse
import os
import numpy as np
import time


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


def generate_new_instance(B, L, D, scores, libraries, scale_factor, noise, rng):
    """Generates a new instance by scaling and perturbing the seed instance."""
    # Apply noise to the global dimensions too, so each instance is different
    new_B = max(1, int(B * scale_factor * rng.uniform(1.0 - noise, 1.0 + noise)))
    new_L = max(1, int(L * scale_factor * rng.uniform(1.0 - noise, 1.0 + noise)))
    new_D = max(1, int(D * scale_factor * rng.uniform(1.0 - noise, 1.0 + noise)))

    # Generate new scores using vectorized numpy operations
    orig_scores = np.array(scores, dtype=np.float64)
    # Tile scores to fill new_B, then add noise
    tiled = np.tile(orig_scores, (new_B // B) + 1)[:new_B]
    noise_vals = rng.uniform(1.0 - noise, 1.0 + noise, size=new_B)
    new_scores = np.maximum(1, (tiled * noise_vals).astype(np.int64))

    new_libraries = []
    for i in range(new_L):
        orig_lib = libraries[i % L]

        # Perturb signup and ship rate
        new_signup = max(1, int(orig_lib['signup'] * rng.uniform(1.0 - noise, 1.0 + noise)))
        new_ship_rate = max(1, int(orig_lib['ship_rate'] * rng.uniform(1.0 - noise, 1.0 + noise)))

        # Determine target number of books (also perturbed with noise)
        target_n_books = max(1, min(
            int(orig_lib['n_books'] * scale_factor * rng.uniform(1.0 - noise, 1.0 + noise)),
            new_B
        ))

        # Fast book selection: just sample target_n_books from [0, new_B)
        new_books = rng.choice(new_B, size=target_n_books, replace=False)

        new_libraries.append({
            'n_books': target_n_books,
            'signup': new_signup,
            'ship_rate': new_ship_rate,
            'books': new_books.tolist()
        })

    return new_B, new_L, new_D, new_scores.tolist(), new_libraries


def main():
    parser = argparse.ArgumentParser(description="Generate new HashCode 2020 Book Scanning instances from seeds.")
    parser.add_argument('seed_file', type=str, help="Path to the seed instance file")
    parser.add_argument('--count', type=int, default=1, help="Number of instances to generate")
    parser.add_argument('--scale', type=float, default=1.0, help="Scale factor for size")
    parser.add_argument('--noise', type=float, default=0.1, help="Noise factor (e.g. 0.1 for 10%% perturbation)")
    parser.add_argument('--out_dir', type=str, default='generated_instances', help="Output directory")
    parser.add_argument('--seed', type=int, default=None, help="Random seed for reproducibility")

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"Reading seed: {args.seed_file}")
    B, L, D, scores, libraries = read_instance(args.seed_file)
    print(f"Original: {B} books, {L} libraries, {D} days")

    base_name = os.path.basename(args.seed_file).split('.')[0]

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    for i in range(args.count):
        t0 = time.time()
        nB, nL, nD, nScores, nLibs = generate_new_instance(
            B, L, D, scores, libraries, args.scale, args.noise, rng
        )

        out_name = f"{base_name}_s{args.scale}_n{args.noise}_{i+1}.txt"
        out_path = os.path.join(args.out_dir, out_name)

        write_instance(out_path, nB, nL, nD, nScores, nLibs)
        elapsed = time.time() - t0
        print(f"[{i+1}/{args.count}] {out_path}  ({nB} books, {nL} libs, {nD} days) — {elapsed:.1f}s")


if __name__ == "__main__":
    main()

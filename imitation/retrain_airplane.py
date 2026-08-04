"""Run the lean Stage A expert, training, evaluation, and promotion workflow."""

import argparse
import os
import subprocess
import sys


def run(command):
    print(">", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--training-seeds", default="42")
    parser.add_argument("--evaluation-seeds", type=int, default=5)
    parser.add_argument("--skip-replay", action="store_true")
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(root, "airplane")
    python = sys.executable

    run([
        python,
        "-u",
        os.path.join(root, "evaluate_airplane.py"),
        "--expert-only",
        "--seeds", str(args.evaluation_seeds),
    ])
    run([
        python,
        "-u",
        os.path.join(root, "collect_data.py"),
        "--scene", "airplane",
        "--episodes", str(args.episodes),
        "--candidate",
        "--headless",
    ])
    run([
        python,
        "-u",
        os.path.join(root, "train_bc.py"),
        "--scene", "airplane",
        "--epochs", str(args.epochs),
        "--seeds", args.training_seeds,
        "--candidate",
    ])
    run([
        python,
        "-u",
        os.path.join(root, "evaluate_airplane.py"),
        "--seeds", str(args.evaluation_seeds),
        "--promote",
    ])

    if not args.skip_replay:
        run([
            python,
            "-u",
            os.path.join(root, "inference.py"),
            "--scene", "airplane",
            "--count", "102",
            "--seed", "0",
            "--export-json", os.path.join(data_dir, "airplane-102.json"),
            "--max-frames", "20000",
        ])


if __name__ == "__main__":
    main()

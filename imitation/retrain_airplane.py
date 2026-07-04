"""Run the candidate-data, multi-seed training, validation, and promotion workflow."""

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
    parser.add_argument("--training-seeds", default="42,43,44")
    parser.add_argument("--evaluation-seeds", type=int, default=20)
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    python = sys.executable
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


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import os
import pickle
import random
import argparse
import numpy as np
import matplotlib.pyplot as plt

FEATURE_DIM = 6

def bucket_bounds(value, boundaries):
    """Return the lower and upper bounds of the bucket that value falls into."""
    if not boundaries:
        return (-np.inf, np.inf)
    idx = np.searchsorted(boundaries, value, side='right')
    if idx == 0:
        return (-np.inf, boundaries[0])
    elif idx == len(boundaries):
        return (boundaries[-1], np.inf)
    else:
        return (boundaries[idx - 1], boundaries[idx])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", required=True, help="Path to the train data file (6col-20rtt-train.p)")
    parser.add_argument("--boundaries-file", required=True, help="Path to the pickle file with bucket boundaries")
    args = parser.parse_args()

    # Load train data
    with open(args.train_file, "rb") as f:
        train_data = pickle.load(f)
    assert train_data.ndim == 3 and train_data.shape[2] == FEATURE_DIM

    # Load boundaries
    with open(args.boundaries_file, "rb") as f:
        boundaries_dict = pickle.load(f)

    # Pick a random sample
    sample_idx = random.randint(0, train_data.shape[0] - 1)
    sample = train_data[sample_idx]  # shape (20, 6)

    # Plot for features 1..5 (i.e., skip feature 0)
    time_points = np.arange(20)
    for feat_sub in range(5):
        feat_idx = feat_sub + 1
        values = sample[:, feat_idx]
        boundaries = boundaries_dict.get(feat_idx, [])

        lower_bounds = []
        upper_bounds = []
        for val in values:
            low, high = bucket_bounds(val, boundaries)
            lower_bounds.append(low)
            upper_bounds.append(high)

        # Plot raw value
        plt.figure(figsize=(8, 4))
        plt.plot(time_points, values, marker='o', label=f"Feature {feat_idx} Value")
        plt.fill_between(time_points, lower_bounds, upper_bounds, color='gray', alpha=0.3, label='Bin Range')
        plt.title(f"Feature {feat_idx} Over Time (Sample {sample_idx})")
        plt.xlabel("Timestep")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"feature_{feat_idx}_sample_{sample_idx}.png")
if __name__ == "__main__":
    main()

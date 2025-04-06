import os
import pickle
import numpy as np
from glob import glob
import argparse

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

DATASET_DIR = "/datastor1/janec/datasets"
BOUNDARY_SAVE_PATH = "/datastor1/janec/datasets/bucket_boundaries.pkl"
FEATURE_DIM = 6

###############################################
#             Collect Feature Values
###############################################
def collect_feature_values(files):
    """
    Collect data for each sample, each feature.
    Returns: list of length FEATURE_DIM; each entry is a Python list of values.
    """
    feature_values = [[] for _ in range(FEATURE_DIM)]
    for f in files:
        with open(f, 'rb') as handle:
            data = pickle.load(handle)  # shape: (N, 20, 6)
            if len(data.shape) != 3 or data.shape[1] < 10 or data.shape[2] != FEATURE_DIM:
                continue

            for i in range(FEATURE_DIM):
                # Flatten the array to 1D
                vals = data[:, :, i].flatten()
                feature_values[i].extend(vals)
    return feature_values

###############################################
#           Binning Helper Functions
###############################################
def get_quantile_boundaries(arr, num_buckets):
    """Quantile-based bucket edges, removing duplicates."""
    percentiles = np.linspace(0, 100, num_buckets + 1)[1:-1]
    bucket_edges = np.percentile(arr, percentiles)
    
    # Convert to a list & remove duplicates
    unique_edges = sorted(set(bucket_edges))
    return unique_edges

def get_freedman_diaconis_boundaries(arr):
    """
    Freedman–Diaconis formula for bin width:
      bin_width = 2 * IQR / cbrt(N)
    => num_bins = (max - min) / bin_width
    This can create a variable number of bins.
    """
    arr = np.sort(arr)
    n = len(arr)
    if n < 2:
        return []

    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    bin_width = 2.0 * iqr / (n ** (1.0/3.0))
    if bin_width <= 0:
        return []

    data_min, data_max = arr[0], arr[-1]
    num_bins = int(np.ceil((data_max - data_min) / bin_width))
    if num_bins < 1:
        num_bins = 1

    boundaries = []
    for b in range(1, num_bins):
        boundary = data_min + b * bin_width
        if boundary < data_max:
            boundaries.append(boundary)
        else:
            break
    return boundaries

###############################################
#      K-Means: 1) Find Optimal K, 2) Boundaries
###############################################
from sklearn.cluster import KMeans

def find_optimal_k_elbow_silhouette(data, max_k=10):
    """
    data: 1D or 2D NumPy array of shape (N, 1) or (N, D).
    max_k: largest number of clusters to try.
    Returns: (best_k_elbow, best_k_silhouette).
    """
    inertia_values = []
    silhouette_values = []
    ks = range(2, max_k + 1)

    best_k_sil = 2
    best_sil_score = -1

    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=42).fit(data)
        inertia_values.append(kmeans.inertia_)

        labels = kmeans.labels_
        sil = silhouette_score(data, labels)
        silhouette_values.append(sil)

        if sil > best_sil_score:
            best_sil_score = sil
            best_k_sil = k

    # "Elbow": pick k with largest drop in inertia (naive approach).
    diffs = np.diff(inertia_values)
    elbow_idx = np.argmax(diffs) 
    best_k_elbow = ks[elbow_idx]

    # Optional: plot them if you'd like
    plt.figure()
    plt.plot(ks, inertia_values, marker='o')
    plt.title("Elbow Method (Inertia)")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.show()
    plt.savefig("elbow_method.png")

    plt.figure()
    plt.plot(ks, silhouette_values, marker='o')
    plt.title("Silhouette Scores")
    plt.xlabel("k")
    plt.ylabel("Average Silhouette Score")
    plt.show()
    plt.savefig("silhouette_scores.png")

    return best_k_elbow, best_k_sil

def get_kmeans_boundaries(arr, n_clusters):
    """
    1D k-means with n_clusters -> boundaries at midpoints between sorted cluster centers.
    """
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is not installed; cannot do K-means.")
    arr_reshaped = arr.reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(arr_reshaped)
    centers = np.sort(kmeans.cluster_centers_.flatten())

    # For k cluster centers, we get k-1 boundaries as midpoints
    boundaries = []
    for c1, c2 in zip(centers[:-1], centers[1:]):
        midpoint = (c1 + c2) / 2.0
        boundaries.append(midpoint)
    return boundaries


###############################################
#            Compute Bucket Boundaries
###############################################
def compute_bucket_boundaries(feature_values, method, num_buckets):
    """
    For each feature i, produce a list of boundary values based on chosen method.
    If method == 'kmeans', we ignore num_buckets for final clusters
      and pick best_k_sil for each feature from find_optimal_k_elbow_silhouette.
    """
    boundaries = {}
    for i, values in enumerate(feature_values):
        # Feature 1 is constant => skip
        if i == 0:
            # Skip feature 1 (base RTT)
            continue

        arr = np.array(values)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            boundaries[i] = []
            continue

        if method == "quantile":
            edges = get_quantile_boundaries(arr, num_buckets)
        elif method == "histogram":
            edges = get_freedman_diaconis_boundaries(arr)
        elif method == "kmeans":
            # Step 1: find best silhouette-based cluster count
            #         with max_k = num_buckets
            arr_reshaped = arr.reshape(-1,1)
            _, best_k_sil = find_optimal_k_elbow_silhouette(arr_reshaped, max_k=num_buckets)
            # Step 2: compute boundaries with that cluster count
            edges = get_kmeans_boundaries(arr, best_k_sil)
        else:
            raise ValueError(f"Unknown bucket method: {method}")

        boundaries[i] = edges

    return boundaries

###############################################
#                     Main
###############################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find bucket boundaries for first 10 RTT features")
    parser.add_argument("--dataset-dir", default=DATASET_DIR, help="Path to directory of .p files")
    parser.add_argument("--save-path", default=BOUNDARY_SAVE_PATH, help="Base path to save boundary pickle")
    parser.add_argument("--method", choices=["quantile","histogram","kmeans"], default="quantile",
                        help="Bucketization method to use")
    parser.add_argument("--num-buckets", type=int, default=1000,
                        help="Max number of buckets (quantile) or max_k (kmeans)")

    args = parser.parse_args()

    # Gather all .p files
    all_pickle_files = glob(os.path.join(args.dataset_dir, "*.p"))
    print(f"Found {len(all_pickle_files)} pickle files in {args.dataset_dir}.")

    # Collect feature values from first 10 RTTs
    features = collect_feature_values(all_pickle_files)

    # Compute boundaries
    bucket_boundaries = compute_bucket_boundaries(features, args.method, args.num_buckets)

    # Adjust final save path with method info
    base, ext = os.path.splitext(args.save_path)
    if args.method == "quantile":
        # e.g., bucket_boundaries-quantile10.pkl
        final_save_path = f"{base}-{args.method}{args.num_buckets}{ext}"
    else:
        # e.g., bucket_boundaries-kmeans.pkl or bucket_boundaries-histogram.pkl
        final_save_path = f"{base}-{args.method}{ext}"

    # Ensure directory exists
    os.makedirs(os.path.dirname(final_save_path), exist_ok=True)

    # Save boundaries
    with open(final_save_path, "wb") as f:
        pickle.dump(bucket_boundaries, f)

    print(f"Bucket boundaries saved to: {final_save_path}")
    for i, b in bucket_boundaries.items():
        print(f"Feature {i} boundaries: {b}")

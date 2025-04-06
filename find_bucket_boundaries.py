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
COMBINED_DIR = "/datastor1/janec/datasets/combined"
BOUNDARY_DIR = "/datastor1/janec/datasets/boundaries"

TRAIN_DATA_PATH = os.path.join(COMBINED_DIR, "6col-20rtt-train.p")
TEST_DATA_PATH = os.path.join(COMBINED_DIR, "6col-20rtt-test.p")

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
            data = pickle.load(handle)  # shape: (N, 20, 6) expected
            if not isinstance(data, np.ndarray) or len(data.shape) != 3 or data.shape[2] != FEATURE_DIM:
                continue

            # Flatten all 20 RTTs for each sample
            for i in range(FEATURE_DIM):
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

    # Example: you can show plots here if desired
    plt.figure()
    plt.plot(ks, inertia_values, marker='o')
    plt.title("Elbow Method (Inertia)")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    plt.savefig("elbow_method.png")
    plt.close()

    plt.figure()
    plt.plot(ks, silhouette_values, marker='o')
    plt.title("Silhouette Scores")
    plt.xlabel("k")
    plt.ylabel("Average Silhouette Score")
    plt.savefig("silhouette_scores.png")
    plt.close()

    return best_k_elbow, best_k_sil

def get_kmeans_boundaries(arr, n_clusters):
    """
    1D k-means with n_clusters -> boundaries at midpoints between sorted cluster centers.
    """
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
    If method == 'kmeans', we pick best silhouette-based cluster count for each feature 
    from find_optimal_k_elbow_silhouette.
    """
    boundaries = {}
    for i, values in enumerate(feature_values):
        # If your Feature 1 is constant => optionally skip if you want:
        if i == 0:
        #     boundaries[i] = []
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
            if not SKLEARN_AVAILABLE:
                raise RuntimeError("scikit-learn is not installed; cannot do K-means.")
            arr_reshaped = arr.reshape(-1,1)
            _, best_k_sil = find_optimal_k_elbow_silhouette(arr_reshaped, max_k=num_buckets)
            edges = get_kmeans_boundaries(arr, best_k_sil)
        else:
            raise ValueError(f"Unknown bucket method: {method}")

        boundaries[i] = edges

    return boundaries

###############################################
#         Train/Test Split Loading
###############################################
def load_or_create_train_test(dataset_dir):
    """
    Check if 6col-20rtt-train.p and 6col-20rtt-test.p exist in:
      /datastor1/janec/datasets/combined/
    If they do, return them.
    Otherwise, gather from all .p files in dataset_dir,
    randomly shuffle, do 80/20 split, and save them.
    Returns list of file paths [train_file, test_file].
    """
    os.makedirs(COMBINED_DIR, exist_ok=True)
    train_path = TRAIN_DATA_PATH
    test_path = TEST_DATA_PATH

    if os.path.exists(train_path) and os.path.exists(test_path):
        print(f"Train/test files found:\n  {train_path}\n  {test_path}")
        return [train_path, test_path]
    else:
        print("No existing train/test split found. Creating new split...")

        all_pickle_files = glob(os.path.join(dataset_dir, "*.p"))
        print(f"Found {len(all_pickle_files)} .p files in {dataset_dir}...")

        # Collect all data into a single list of arrays
        all_data_arrays = []
        for f in all_pickle_files:
            with open(f, 'rb') as handle:
                data = pickle.load(handle)
                # Expect (N, 20, 6)
                if (
                    isinstance(data, np.ndarray) 
                    and len(data.shape) == 3 
                    and data.shape[2] == FEATURE_DIM
                ):
                    all_data_arrays.append(data)
        if not all_data_arrays:
            print("No valid .p data found.")
            return []

        combined_data = np.concatenate(all_data_arrays, axis=0)  # shape (X, 20, 6)
        print(f"Combined dataset shape: {combined_data.shape}")

        # Shuffle
        np.random.shuffle(combined_data)

        # Split 80/20
        cutoff = int(0.8 * len(combined_data))
        train_data = combined_data[:cutoff]
        test_data = combined_data[cutoff:]

        with open(train_path, "wb") as f:
            pickle.dump(train_data, f)
        with open(test_path, "wb") as f:
            pickle.dump(test_data, f)

        print(f"Created train: {train_path} -> shape {train_data.shape}")
        print(f"Created test : {test_path} -> shape {test_data.shape}")

        return [train_path, test_path]

###############################################
#                     Main
###############################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find bucket boundaries for (N, 20, 6) data, with train/test split")
    parser.add_argument("--dataset-dir", default=DATASET_DIR, help="Path to directory of .p files")
    parser.add_argument("--method", choices=["quantile","histogram","kmeans"], default="quantile",
                        help="Bucketization method to use")
    parser.add_argument("--num-buckets", type=int, default=1000,
                        help="Max number of buckets (quantile) or max_k (kmeans)")

    args = parser.parse_args()

    # 1) Load or create train/test .p files
    split_files = load_or_create_train_test(args.dataset_dir)
    if len(split_files) < 2:
        print("No data to process. Exiting.")
        exit(0)

    # 2) Combine the train/test for boundary calculation 
    #    (Alternatively, you can just read from train data to avoid leakage.)
    data_files = split_files  # or [split_files[0]] if you want train-only

    # 3) Collect feature values
    feature_values = collect_feature_values(data_files)

    # 4) Compute boundaries
    bucket_boundaries = compute_bucket_boundaries(feature_values, args.method, args.num_buckets)

    # 5) Save boundaries under /datastore1/janec/datasets/boundaries
    # os.makedirs(BOUNDARY_DIR, exist_ok=True)

    # e.g. boundaries-quantile1000.pkl or boundaries-kmeans.pkl
    if args.method == "quantile":
        boundaries_filename = f"boundaries-{args.method}{args.num_buckets}.pkl"
    else:
        boundaries_filename = f"boundaries-{args.method}.pkl"
    final_save_path = os.path.join(BOUNDARY_DIR, boundaries_filename)

    with open(final_save_path, "wb") as f:
        pickle.dump(bucket_boundaries, f)

    print(f"Bucket boundaries saved to: {final_save_path}")
    for i, b in bucket_boundaries.items():
        print(f"Feature {i} boundaries: {b}")

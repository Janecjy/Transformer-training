import os
import pickle
import numpy as np

# ----------------------------
# 1) LOAD DATA
# ----------------------------
dataset_path = "NEWDatasets/ccbench-dataset-raw/6col-rtt-based.p"
with open(dataset_path, "rb") as f:
    dataset = pickle.load(f)  # Expect shape (N, 20, 6)

print("Loaded dataset shape:", dataset.shape)  # (N, 20, 6)

# ----------------------------
# 2) DEFINE BUCKET BOUNDARIES
#    AND WHICH FEATURES TO BUCKETIZE
# ----------------------------
# Feature 0 = baseRTT (normalize)
# Features 1..5 = other metrics (bucketize)

bucket_boundaries_ccbench = {
    1: [0.12, 0.2, 0.28, 0.43, 0.55, 0.83, 1.03, 1.63, 2.12, 4.02, 8, 12],
    2: [0.01, 0.3, 0.38, 0.44, 0.49, 0.54, 0.6, 0.68, 0.84, 1.41, 3, 5],
    3: [0.08, 0.11, 0.15, 0.23, 0.45, 0.8, 0.9, 1, 1.75],
    4: [0.0002, 0.0047, 0.0361, 0.1, 0.2, 0.3],
    5: [0.75, 1, 1.001, 1.003, 1.012, 1.25]
}

# ----------------------------
# 3) MIN-MAX NORMALIZE BASE RTT (feature #0)
#    across all (N*20) values
# ----------------------------
N, T, F = dataset.shape  # (N, 20, 6)

base_rtt_vals = dataset[..., 0].reshape(-1)  # Flatten (N*20,)
rtt_min = base_rtt_vals.min()
rtt_max = base_rtt_vals.max()

if np.isclose(rtt_min, rtt_max):
    # Avoid divide-by-zero if all RTT are the same
    print("Warning: base RTT range is zero; forcing all to 0.0")
    dataset[..., 0] = 0.0
else:
    scaled = (base_rtt_vals - rtt_min) / (rtt_max - rtt_min)
    dataset[..., 0] = scaled.reshape(N, T)

# ----------------------------
# 4) BUCKETIZE FEATURES (1..5)
#    Each feature has its own boundary => own vocabulary
# ----------------------------

feature_ids = [1,2,3,4,5]
bin_offsets = {}
running_offset = 0

for feat_idx in feature_ids:
    boundaries = bucket_boundaries_ccbench[feat_idx]
    # number of bins = len(boundaries) + 1
    bin_offsets[feat_idx] = running_offset
    running_offset += (len(boundaries) + 1)

def bucketize_with_offset(values_1d, boundaries, offset):
    # discrete bin in [0..len(boundaries)]
    bins_local = np.searchsorted(boundaries, values_1d, side='right')
    # shift by 'offset' => bins in [offset .. offset+len(boundaries)]
    return bins_local + offset

for feat_idx in feature_ids:
    col_vals = dataset[..., feat_idx].reshape(-1)
    boundaries = bucket_boundaries_ccbench[feat_idx]
    offset = bin_offsets[feat_idx]
    discrete_bins = bucketize_with_offset(col_vals, boundaries, offset)
    dataset[..., feat_idx] = discrete_bins.reshape(N, T)

print("Example of one sample after normalization/bucketization:")
print(dataset[0])

# ----------------------------
# 5) SPLIT 80/20 TRAIN/TEST
#    (Split by sample dimension N)
# ----------------------------
num_samples = N
train_size = int(0.8 * num_samples)

indices = np.random.permutation(num_samples)
train_idx = indices[:train_size]
test_idx  = indices[train_size:]

train_data = dataset[train_idx]  # (train_size, 20, 6)
test_data  = dataset[test_idx]   # (num_samples-train_size, 20, 6)

print("Train data shape:", train_data.shape)
print("Test  data shape:", test_data.shape)

# ----------------------------
# 6) SAVE PREPROCESSED ARRAYS
# ----------------------------
out_dir = "NEWDatasets/ccbench-dataset-preprocessed"
os.makedirs(out_dir, exist_ok=True)

train_path = os.path.join(out_dir, "6col-rtt-based-train.p")
test_path  = os.path.join(out_dir, "6col-rtt-based-test.p")

with open(train_path, "wb") as f:
    pickle.dump(train_data, f)

with open(test_path, "wb") as f:
    pickle.dump(test_data, f)

print(f"Saved train set to: {train_path}")
print(f"Saved test  set to: {test_path}")

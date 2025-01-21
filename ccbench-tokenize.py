import os
import pickle
import numpy as np

# ----------------------------
# 1) LOAD DATA
# ----------------------------
dataset_path = "NEWDatasets/ccbench-dataset-raw/6col-rtt-based.p"
with open(dataset_path, "rb") as f:
    ccbench_dataset = pickle.load(f)  # Expect shape (N, 20, 6)
genet_data1 = np.load('NEWDatasets/genet-dataset-raw/tcp_metrics.npy')
genet_data2 = np.load('NEWDatasets/genet-dataset-raw/tcp_metrics_trace_file_2.npy')

# --- Function to reshape data to (M, 20, 6) and ignore leftover rows ---
def reshape_2d_to_3d_20(data_2d):
    """
    Reshape a 2D NumPy array of shape (N, 6) into (M, 20, 6).
    Ignores leftover rows if N is not divisible by 20.
    """
    # Number of full 20-row segments
    num_segments = data_2d.shape[0] // 20
    
    # Truncate data so it has exactly num_segments*20 rows
    truncated = data_2d[:num_segments * 20, :]
    
    # Reshape to (M, 20, 6)
    data_3d = truncated.reshape(num_segments, 20, 6)
    
    return data_3d

# --- Apply the reshape function to both genet_data1 and genet_data2 ---
genet_data1_3d = reshape_2d_to_3d_20(genet_data1)
genet_data2_3d = reshape_2d_to_3d_20(genet_data2)

print("\nReshaped shapes:")
print("genet_data1_3d:", genet_data1_3d.shape)
print("genet_data2_3d:", genet_data2_3d.shape)

combined_data = np.vstack([genet_data1_3d, genet_data2_3d, ccbench_dataset])
print("combined_data shape:", combined_data.shape)

# print("Loaded dataset shape:", dataset.shape)  # (N, 20, 6)

# ----------------------------
# 2) DEFINE BUCKET BOUNDARIES
#    AND WHICH FEATURES TO BUCKETIZE
# ----------------------------
# Feature 0 = baseRTT (normalize)
# Features 1..5 = other metrics (bucketize)

bucket_boundaries_ccbench = {
    1: [0.12, 0.2, 0.28, 0.43, 0.55, 0.83, 1.03, 1.29, 1.63, 2.12, 3.64, 4.02, 5.74, 8, 12, 14],
    2: [0.01, 0.3, 0.38, 0.44, 0.49, 0.54, 0.6, 0.68, 0.84, 1.41, 3, 5, 205, 395, 1206],
    3: [0.01, 0.08, 0.11, 0.15, 0.23, 0.45, 0.8, 0.9, 1, 1.75],
    4: [0.0002, 0.0047, 0.0361, 0.1, 0.2, 0.3],
    5: [0.75, 1, 1.001, 1.003, 1.012, 1.25, 3.52, 4.7, 5.39, 6.26]
}

# ----------------------------
# 3) MIN-MAX NORMALIZE BASE RTT (feature #0)
#    across all (N*20) values
# ----------------------------
N, T, F = combined_data.shape  # (N, 20, 6)

base_rtt_vals = combined_data[..., 0].reshape(-1)  # Flatten (N*20,)
rtt_min = base_rtt_vals.min()
rtt_max = base_rtt_vals.max()
print("rtt_min: ", rtt_min)
print("rtt_max: ", rtt_max)

if np.isclose(rtt_min, rtt_max):
    # Avoid divide-by-zero if all RTT are the same
    print("Warning: base RTT range is zero; forcing all to 0.0")
    combined_data[..., 0] = 0.0
else:
    scaled = (base_rtt_vals - rtt_min) / (rtt_max - rtt_min)
    combined_data[..., 0] = scaled.reshape(N, T)

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
    col_vals = combined_data[..., feat_idx].reshape(-1)
    boundaries = bucket_boundaries_ccbench[feat_idx]
    offset = bin_offsets[feat_idx]
    discrete_bins = bucketize_with_offset(col_vals, boundaries, offset)
    combined_data[..., feat_idx] = discrete_bins.reshape(N, T)

print("Example of one sample after normalization/bucketization:")
print(combined_data[0])

# ----------------------------
# 5) SPLIT 80/20 TRAIN/TEST
#    (Split by sample dimension N)
# ----------------------------
num_samples = N
train_size = int(0.8 * num_samples)

indices = np.random.permutation(num_samples)
train_idx = indices[:train_size]
test_idx  = indices[train_size:]

train_data = combined_data[train_idx]  # (train_size, 20, 6)
test_data  = combined_data[test_idx]   # (num_samples-train_size, 20, 6)

print("Train data shape:", train_data.shape)
print("Test  data shape:", test_data.shape)

# ----------------------------
# 6) SAVE PREPROCESSED ARRAYS
# ----------------------------
out_dir = "NEWDatasets/combined-dataset-preprocessed"
os.makedirs(out_dir, exist_ok=True)

train_path = os.path.join(out_dir, "6col-rtt-based-train.p")
test_path  = os.path.join(out_dir, "6col-rtt-based-test.p")

with open(train_path, "wb") as f:
    pickle.dump(train_data, f)

with open(test_path, "wb") as f:
    pickle.dump(test_data, f)

print(f"Saved train set to: {train_path}")
print(f"Saved test  set to: {test_path}")

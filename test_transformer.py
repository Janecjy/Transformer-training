import torch
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from models import Seq2SeqWithEmbeddingmodClassMultiHead
from utils import create_mask
from tqdm import tqdm
import bisect

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
PAD_IDX = 2
PREDICTION_LENGTH = 10

# === CONFIG ===
TEST_SET_PATH = "/datastor1/janec/datasets_old/combined/6col-20rtt-test.p"
BOUNDARIES_MAP = {
    "quantile50": "/datastor1/janec/datasets_old/combined/boundaries-quantile50-tokenized-multi.pkl",
    "quantile100": "/datastor1/janec/datasets_old/combined/boundaries-quantile100-tokenized-multi.pkl",
}
MODELS_TO_TEST = [
    "/datastor1/janec/models_old/Checkpoint-Combined_10RTT_6col_Transformer3_128_2_2_16_2_lr_0.0001_boundaries-quantile50_multi-619iter.p",
    "/datastor1/janec/models_old/Checkpoint-Combined_10RTT_6col_Transformer3_256_4_4_32_4_lr_0.0001_boundaries-quantile100_multi-369iter.p",
    "/datastor1/janec/models_old/Checkpoint-Combined_10RTT_6col_Transformer3_256_3_3_32_4_lr_5e-05_boundaries-quantile100_multi-419iter.p",
    "/datastor1/janec/models_old/Checkpoint-Combined_10RTT_6col_Transformer3_256_4_4_32_4_lr_0.0001_boundaries-quantile50_multi-399iter.p",
]
NUM_FEATURES = 5  # exclude base RTT

# === LOAD TEST SET ===
with open(TEST_SET_PATH, "rb") as f:
    test_data_np = pickle.load(f)
    # sample
    test_data_np = test_data_np[:30000]  # for testing
test_data = torch.tensor(test_data_np, dtype=torch.float32).to(DEVICE)

# === HELPER FUNCTIONS ===
def compute_cdf(distances):
    sorted_vals = np.sort(distances)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    return sorted_vals, cdf

def get_true_bucket_index(value, boundaries):
    # returns index 0 to len(boundaries)
    return bisect.bisect_right(boundaries, value)

def make_bucket_midpoints(boundaries):
    midpoints = []

    # Leftmost bucket: extrapolate midpoint below first boundary
    left_gap = boundaries[1] - boundaries[0] if len(boundaries) > 1 else 1
    leftmost = boundaries[0] - left_gap / 2
    midpoints.append(leftmost)

    # Middle buckets
    for i in range(len(boundaries) - 1):
        mid = (boundaries[i] + boundaries[i+1]) / 2
        midpoints.append(mid)

    # Rightmost bucket: extrapolate
    right_gap = boundaries[-1] - boundaries[-2] if len(boundaries) > 1 else 1
    rightmost = boundaries[-1] + right_gap / 2
    midpoints.append(rightmost)

    return midpoints

# === MAIN TEST LOOP ===
cdf_val_results = {}
cdf_bucket_dist_results = {}

for model_path in MODELS_TO_TEST:
    # Determine boundary file from model name
    if "quantile50" in model_path:
        boundaries_path = BOUNDARIES_MAP["quantile50"]
    elif "quantile100" in model_path:
        boundaries_path = BOUNDARIES_MAP["quantile100"]
    else:
        raise ValueError(f"Unknown quantile for model: {model_path}")

    with open(boundaries_path, "rb") as f:
        boundary_data = pickle.load(f)
    boundaries_dict = boundary_data["boundaries_dict"]

    # Compute bucket midpoints (include extrapolated rightmost bucket)
    bucket_midpoints = {}
    for feat, boundaries in boundaries_dict.items():
        bucket_midpoints[feat] = make_bucket_midpoints(boundaries)
        # print(f"Feature {feat}: {len(bucket_midpoints[feat])} midpoints (for {len(boundaries)+1} buckets)")

    # print(f"\nLoading model: {model_path}")
    model = torch.load(model_path, map_location=DEVICE)
    model.eval()

    all_val_distances = []
    all_bucket_distances = []

    with torch.no_grad():
        for i in tqdm(range(test_data.shape[0])):
            sample = test_data[i:i+1]  # shape (1, 20, 6)
            enc_input = sample[:, :-PREDICTION_LENGTH, :]
            dec_input = 1.5 * torch.ones((1, PREDICTION_LENGTH, 6)).to(DEVICE)
            expected_output = sample[:, -PREDICTION_LENGTH:, :]

            src_mask, tgt_mask, _, _ = create_mask(enc_input, dec_input, PAD_IDX, DEVICE)
            pred = model(enc_input, dec_input, src_mask, tgt_mask, None, None, None)  # shape (1, 10, 5, B)

            for t in range(PREDICTION_LENGTH):
                for feat in range(NUM_FEATURES):
                    feat_idx = feat + 1  # skip base RTT

                    bucket_idx = pred[0, t, feat].argmax().item()
                    midpoints = bucket_midpoints[feat + 1]
                    if bucket_idx >= len(midpoints):
                        print(f"[WARNING] bucket_idx {bucket_idx} out of range for feature {feat+1}. Clamping.")
                        bucket_idx = len(midpoints) - 1

                    predicted_value = midpoints[bucket_idx]
                    true_value = expected_output[0, t, feat + 1].item()

                    bucket_idx = pred[0, t, feat].argmax().item()
                    # print(f"Predicted bucket index for feature {feat_idx}: {bucket_idx}")
                    predicted_value = bucket_midpoints[feat_idx][bucket_idx]
                    true_value = expected_output[0, t, feat_idx].item()
                    val_dist = abs(predicted_value - true_value)

                    # Get true bucket index from boundaries
                    boundaries = boundaries_dict[feat_idx]
                    true_bucket = get_true_bucket_index(true_value, boundaries)
                    bucket_dist = abs(bucket_idx - true_bucket)

                    all_val_distances.append(val_dist)
                    all_bucket_distances.append(bucket_dist)

    # CDFs
    model_name_short = os.path.basename(model_path)
    cdf_val_results[model_name_short] = compute_cdf(all_val_distances)
    cdf_bucket_dist_results[model_name_short] = compute_cdf(all_bucket_distances)

pickle.dump(cdf_val_results, open("cdf_val_results.pkl", "wb"))
pickle.dump(cdf_bucket_dist_results, open("cdf_bucket_dist_results.pkl", "wb"))

# === PLOT VALUE CDF ===
plt.figure(figsize=(10, 6))
for model_name, (distances, cdf) in cdf_val_results.items():
    plt.plot(distances, cdf, label=model_name)
plt.title("CDF of Absolute Distance (Predicted Bucket Midpoint vs True Value)")
plt.xlabel("Absolute Value Distance")
plt.ylabel("CDF")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("model_cdf_value_distance.png")
plt.show()

# === PLOT BUCKET INDEX DISTANCE CDF ===
plt.figure(figsize=(10, 6))
for model_name, (distances, cdf) in cdf_bucket_dist_results.items():
    plt.plot(distances, cdf, label=model_name)
plt.title("CDF of Bucket Index Distance (Predicted vs True Bucket)")
plt.xlabel("Bucket Index Distance")
plt.ylabel("CDF")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("model_cdf_bucket_distance.png")
plt.show()

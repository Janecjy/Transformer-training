import torch
import pickle
import numpy as np
import time

# Load the dataset
dataset_name = 'FullDataset-new-filtered1'
with open('NEwDatasets-new/FullDataset.p', 'rb') as f:
    d = pickle.load(f)
N = d['normalizer'].detach().cpu().numpy()  # Load normalizer

with open('./NEWDatasets/' + dataset_name + '-test.p', 'rb') as f:
    train_dataset = pickle.load(f)
    # train_dataset = train_dataset[0:1, :, :]  # Adjusted for a single dataset
    # print(train_dataset.shape)

# The bucket boundaries for each feature index
bucket_boundaries = {
    1: [5, 7, 8, 10, 14, 23, 34, 48, 69],
    4: [20, 26, 31, 34, 37, 40, 42, 46, 51, 60],
    6: [1, 2, 3, 5],
    8: [1, 2, 3, 4, 5, 9, 14, 20, 30],
    12: [0.5, 0.6, 1.15, 1.74, 2.32, 2.9, 3.48, 4.64, 6.37, 8.1, 9.85, 11.58, 13.9, 16.22, 19.11, 23.17, 29.54]
}

# Function to assign each value to a bucket
def assign_buckets(values, boundaries):
    # print("values: ", values)
    bucket_counts = np.zeros((values.shape[0], len(boundaries) + 1), dtype=int)
    for i, boundary in enumerate(boundaries):
        bucket_counts[:, i] = (values < boundary).sum(axis=1)
    bucket_counts[:, -1] = (values >= boundaries[-1]).sum(axis=1)
    # print("bucket counts: ", bucket_counts)
    return bucket_counts

# Extract the required features, apply normalizer, and convert to bucket counts
def process_tokens(tokens, bucket_boundaries, normalizer):
    bucket_features = []
    for idx, boundaries in bucket_boundaries.items():
        # Multiply the token values by the normalizer for that index
        feature_values = tokens[:, idx] * normalizer[idx]  # Apply normalization
        bucket_counts = assign_buckets(feature_values.unsqueeze(1), boundaries)
        bucket_features.append(bucket_counts)
    return np.concatenate(bucket_features, axis=1)

# Function to process each sample
def process_sample(sample, normalizer):
    return process_tokens(sample, bucket_boundaries, normalizer)

# Single-threaded processing function
def single_thread_process(train_dataset, normalizer):
    transformed_dataset = []
    
    # Iterate through each sample in the dataset
    for sample in train_dataset:
        transformed_sample = process_sample(sample, normalizer)
        transformed_dataset.append(transformed_sample)
    
    # Convert the list to a torch tensor
    transformed_dataset = torch.tensor(transformed_dataset)
    return transformed_dataset

# Perform single-threaded processing
t0 = time.time()
transformed_dataset = single_thread_process(train_dataset, N)
time_taken = time.time() - t0
print("Time taken: ", time_taken)

print("Transformed dataset shape: ", transformed_dataset.shape)
# print("Transformed dataset: ", transformed_dataset)

# Save the transformed dataset
with open('./NEWDatasets/FullDataset-new-filtered1-bucketized-test.p', 'wb') as f:
    pickle.dump(transformed_dataset, f, pickle.HIGHEST_PROTOCOL)

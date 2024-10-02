import torch
import pickle
import numpy as np
import time

# Load the dataset
dataset_name = 'FullDataset1x-filtered1'
with open('NEWDatasets/'+dataset_name+'.p', 'rb') as f:
    d = pickle.load(f)
train_dataset = d['data']
N = d['normalizer'].detach().cpu().numpy()  # Load normalizer

# The bucket boundaries for each feature index
bucket_boundaries_1x = {
    1: [2, 3, 4, 5, 7, 8, 9, 12, 169],
    4: [20, 42.5, 46, 48.7, 51, 54, 58, 63, 70, 87, 3198],
    6: [1, 2, 3, 4, 5, 12],
    8: [1, 2, 3, 4, 5, 12],
    12: [0.6, 1.16, 1.74, 2.32, 2.9, 3.48, 8.1]
}

bucket_boundaries_10x = {
    1: [3, 5, 8, 15, 24, 34, 46, 60, 81, 743],
    4: [20, 28, 32, 34, 37, 39, 40, 42, 45, 58, 1998],
    6: [1, 2, 3, 4, 6, 11, 408],
    8: [2, 3, 5, 8, 11, 15, 19, 23, 29, 39, 125],
    12: [0.6, 1.16, 1.74, 2.9, 4.63, 6.37, 8.1, 9.3, 11, 12.16, 14, 16.22, 18, 20.27, 23.17, 16.64, 32.44, 70]
}

if '1x' in dataset_name:
    bucket_boundaries = bucket_boundaries_1x
elif '10x' in dataset_name:
    bucket_boundaries = bucket_boundaries_10x
    
# Function to assign each value to a bucket
def assign_buckets(values, boundaries):
    # print("values: ", values)
    bucket_counts = np.zeros((values.shape[0], len(boundaries) + 1), dtype=int)
    for i, boundary in enumerate(boundaries):
        if i > 0:
            bucket_counts[:, i] = ((values < boundary) & (values >= boundaries[i-1])).sum(axis=1)
        else:
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
        # return transformed_dataset
    
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
with open('./NEWDatasets/'+dataset_name+'-bucketized.p', 'wb') as f:
    pickle.dump(transformed_dataset, f, pickle.HIGHEST_PROTOCOL)

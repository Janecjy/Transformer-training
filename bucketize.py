import torch
import pickle
import numpy as np
import time

# Load the dataset
dataset_name = 'FullDataset_alt1x'
with open('NEWDatasets/'+dataset_name+'.p', 'rb') as f:
    d = pickle.load(f)
train_dataset = d['data']
print(train_dataset.shape)
N = d['normalizer'].detach().cpu().numpy()  # Load normalizer

# The bucket boundaries for each feature index
bucket_boundaries_1x = {
    1: [2, 3, 4, 5, 6, 7, 8, 10, 12, 169],
    4: [20, 42.5, 46, 48.7, 51, 54, 58, 63, 70, 87, 3198],
    6: [1, 2, 3, 4, 5, 12],
    8: [1, 2, 3, 4, 5, 13],
    12: [0.6, 1.16, 1.74, 2.32, 2.9, 3.48, 4.05, 16.8]
}

bucket_boundaries_10x = {
    1: [1, 4, 7, 11, 17, 24, 33, 43, 57, 77, 1414],
    4: [20, 25.4, 29.6, 32.5, 35, 37, 39, 41, 43, 50, 1998],
    6: [2, 3, 4, 5, 7, 12, 412],
    8: [2, 3, 5, 7, 10, 13, 17, 21, 27, 37, 126],
    12: [ 0.58, 1.16, 1.74,  2.32,  2.9 ,  3.48,
        4.05,  5.21,  5.79,  6.37,  7.53,  8.11,  8.69,  9.85, 10.43,
       11.58, 12.74, 13.32, 14.48, 16.22, 17.38, 19.11, 20.85, 22.59,
       25.48, 28.96, 35.33, 70.08]
}
torch.set_printoptions(precision=2, sci_mode=False)
if '1x' in dataset_name:
    bucket_boundaries = bucket_boundaries_1x
elif '10x' in dataset_name:
    bucket_boundaries = bucket_boundaries_10x

# Get max bucket count for each feature by comparing 1x and 10x
max_bucket_counts = {
    idx: max(len(bucket_boundaries_1x.get(idx, [])), len(bucket_boundaries_10x.get(idx, []))) + 1
    for idx in set(bucket_boundaries_1x.keys()).union(bucket_boundaries_10x.keys())
}
# print("Max bucket counts: ", max_bucket_counts)
    
# Function to assign each value to a bucket
def assign_buckets(values, boundaries, max_bucket_counts):
    # print("values: ", values)
    bucket_counts = np.zeros((values.shape[0], max_bucket_counts+1), dtype=int)
    for i, boundary in enumerate(boundaries):
        if i > 0:
            bucket_counts[:, i] = ((values < boundary) & (values >= boundaries[i-1])).sum(axis=1)
        else:
            bucket_counts[:, i] = (values < boundary).sum(axis=1)
    bucket_counts[:, len(boundaries)] = (values >= boundaries[-1]).sum(axis=1)
    if len(boundaries) < max_bucket_counts:
        bucket_counts[:, len(boundaries)+1:] = -1
    # print("bucket counts: ", bucket_counts[0, :])
    return bucket_counts

# Extract the required features, apply normalizer, and convert to bucket counts
def process_tokens(tokens, bucket_boundaries, normalizer):
    bucket_features = []
    for idx, boundaries in bucket_boundaries.items():
        # Multiply the token values by the normalizer for that index
        feature_values = tokens[:, idx] * normalizer[idx]  # Apply normalization
        bucket_counts = assign_buckets(feature_values.unsqueeze(1), boundaries, max_bucket_counts[idx])
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
        # sample_value = sample * normalizer
        # print("Sample: ",  sample_value[:, [1, 4, 6, 8, 12]])
        transformed_sample = process_sample(sample, normalizer)
        transformed_dataset.append(transformed_sample)
        # print("Transformed sample: ", transformed_sample)
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

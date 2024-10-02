import pickle
import torch
import numpy as np
from sklearn.model_selection import train_test_split

# Load the dataset
with open('NEWDatasets/FullDataset1x-filtered1-bucketized.p', 'rb') as f:
    dataset = pickle.load(f)

# Get the data tensor (shape: [93886, 64, 44])
data = dataset

# Split into train and test sets (8:2 ratio)
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Save train and test sets
with open('NEWDatasets/FullDataset1x-filtered1-bucketized-train.p', 'wb') as f_train, open('NEWDatasets/FullDataset1x-filtered1-bucketized-test.p', 'wb') as f_test:
    pickle.dump(train_data, f_train)
    pickle.dump(test_data, f_test)

# Create a set of unique 44-size vectors from the data
unique_vectors = set()
for sample in data:
    for vector in sample:
        unique_vectors.add(tuple(vector.tolist()))  # Convert each vector to a tuple to add to the set

# Create a mapping from each unique vector to an index
vocab_dict = {vector: idx for idx, vector in enumerate(unique_vectors)}

# Save the vocabulary dictionary
with open('NEWDatasets/FullDataset1x-filtered1-bucketized-VocabDict.p', 'wb') as f_vocab:
    pickle.dump(vocab_dict, f_vocab)

print("Train/Test sets and vocabulary dictionary created successfully!")

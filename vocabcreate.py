import pickle
import torch
import numpy as np

# ----------------------
# 1) LOAD THE DATA
# ----------------------
with open('NEWDatasets/ccbench-dataset-preprocessed/6col-rtt-based-test.p', 'rb') as f:
    rtt_based_test = pickle.load(f)
with open('NEWDatasets/ccbench-dataset-preprocessed/6col-rtt-based-train.p', 'rb') as f:
    rtt_based_train = pickle.load(f)
with open('NEWDatasets/ccbench-dataset-preprocessed/6col-time-based-test.p', 'rb') as f:
    time_based_test = pickle.load(f)
with open('NEWDatasets/ccbench-dataset-preprocessed/6col-time-based-train.p', 'rb') as f:
    time_based_train = pickle.load(f)

# Concatenate all data along the first (sample) dimension
data = np.concatenate((rtt_based_test, rtt_based_train, time_based_test, time_based_train), axis=0)
print("Combined data shape:", data.shape)
# If shape is (N, T, 6), then data[i, j, :] is (6,).

# ----------------------
# 2) BUILD VOCAB FOR FEATURES #1..5 ONLY
#    (Excluding base RTT at index 0)
# ----------------------
unique_vectors = set()

# data shape is (total_samples, T, 6)
for sample in data:
    for vector in sample:
        # vector is shape (6,)
        # Exclude the first element (base RTT) => keep last 5 features
        discrete_part = vector[1:]  # shape (5,)
        # Convert to tuple so it can be added to a set
        unique_vectors.add(tuple(discrete_part.tolist()))

# Free memory
del data
torch.cuda.empty_cache()

# ----------------------
# 3) CREATE MAPPING FROM UNIQUE VECTOR -> INDEX
# ----------------------
vocab_dict = {vec: idx for idx, vec in enumerate(unique_vectors)}

with open('NEWDatasets/ccbench-dataset-preprocessed/6col-VocabDict.p', 'wb') as f_vocab:
    pickle.dump(vocab_dict, f_vocab, protocol=pickle.HIGHEST_PROTOCOL)


vocab_back_dict = {idx: vec for idx, vec in enumerate(unique_vectors)}

with open('NEWDatasets/ccbench-dataset-preprocessed/6col-VocabBackDict.p', 'wb') as f_vocab_back:
    pickle.dump(vocab_back_dict, f_vocab_back, protocol=pickle.HIGHEST_PROTOCOL)
    

print("Vocabulary dictionary created successfully (excluding base RTT)!")

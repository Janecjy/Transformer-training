import sys
import pickle
import torch

dataset_name = sys.argv[1]
column_index = int(sys.argv[2])
new_dataset_name = dataset_name + '-' + str(column_index) + '-sum'

with open('./NEWDatasets/'+dataset_name+'.p', 'rb') as f:
    train_dataset = pickle.load(f)

    # Replace each value in the 6th column with the sum of the last 10 values
    for i in range(train_dataset.shape[0]):
        for j in range(train_dataset.shape[1]):
            if j >= 10:
                train_dataset[i, j, column_index] = torch.sum(train_dataset[i, j-10:j, column_index])
            else:
                train_dataset[i, j, column_index] = torch.sum(train_dataset[i, :j, column_index])
                
# Save the selected 80k samples
with open('./NEWDatasets/'+new_dataset_name+'.p', 'wb') as f:
    pickle.dump(train_dataset, f, pickle.HIGHEST_PROTOCOL)

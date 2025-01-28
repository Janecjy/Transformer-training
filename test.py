import torch
import pickle
from models import create_mask
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch.nn as nn
# from runvocabmlp import MLP
import sys
import numpy as np

PAD_IDX = 2
DEVICE = 'cpu'
ITER = sys.argv[1]

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512) -> None:
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        if hidden_dim is not None: 
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
                nn.Softmax(dim=1)  # Softmax layer added here
            )
        else:
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, output_dim), 
                nn.ReLU(),
                nn.Softmax(dim=1)  # Softmax layer added here
            )
    
    def forward(self, x):
        x = x.float()
        x = self.mlp(x)
        return x

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
       25.48, 28.96, 35.33, 70.08]}

bucket_boundaries_ccbench = {
    1: [0.12, 0.2, 0.28, 0.43, 0.55, 0.83, 1.03, 1.63, 2.12, 4.02, 8, 12],
    2: [0.01, 0.3, 0.38, 0.44, 0.49, 0.54, 0.6, 0.68, 0.84, 1.41, 3, 5],
    3: [0.08, 0.11, 0.15, 0.23, 0.45, 0.8, 0.9, 1, 1.75],
    4: [0.0002, 0.0047, 0.0361, 0.1, 0.2, 0.3],
    5: [0.75, 1, 1.001, 1.003, 1.012, 1.25]
}


def test_model_accuracy(model_list, model_name_list, is_transformer_list, dataset, vocab_dict, prediction_len, batch_size=32, device='cpu'):
    """
    Test the accuracy of the model by comparing the predicted class (max index in output)
    with the true class from the dataset.
    
    Args:
    - model_list: List of models.
    - is_transformer_list: List of booleans indicating whether the model is a transformer.
    - dataset: Test dataset (tensor).
    - vocab_dict: Dictionary mapping 44-size vectors to class indices.
    - prediction_len: Length of the prediction (number of tokens to predict).
    - batch_size: Batch size for testing.
    - device: Device to run the test on ('cuda' or 'cpu').
    
    Returns:
    - accuracy: The accuracy of the model on the test dataset.
    """
    correct_predictions_dict = {model_name_list[i]: 0 for i in range(len(model_name_list))}
    for i, model in enumerate(model_list):
        model.eval()  # Set model to evaluation mode
    total_predictions = 0
    
    num_batches = dataset.shape[0] // batch_size
    with torch.no_grad():  # Disable gradient computation during testing
        for batch in range(num_batches):
            # Get the batch input
            input_batch = dataset[batch * batch_size:(batch + 1) * batch_size, :, :].clone()
            enc_input = input_batch[:, :-prediction_len, :].to(device)
            dec_input = (1.5 * torch.ones((batch_size, prediction_len, input_batch.shape[2]))).to(device)
            
            src_mask, tgt_mask, _, _ = create_mask(enc_input, dec_input, pad_idx=PAD_IDX, device=device)
            expected_output = input_batch[:, -prediction_len:, :].to(device)
            
            # Convert expected_output to class indices (ground truth)
            batch_classes = []
            for i in range(batch_size):
                for t in range(prediction_len):
                    vector = tuple(expected_output[i, t, :].tolist())
                    class_idx = vocab_dict.get(vector, -1)  # Get class index from vocab_dict
                    if class_idx == -1:
                        raise ValueError(f"Vector not found in vocab_dict: {vector}, expected output: {expected_output[i, t, :]}")
                    batch_classes.append(class_idx)

            # Convert to tensor and move to device
            batch_classes = torch.tensor(batch_classes, dtype=torch.long).to(device)
            batch_classes = batch_classes.view(batch_size, prediction_len)
            
            for i, model in enumerate(model_list):
                if is_transformer_list[i]:
                    # Forward pass through the model to get the predicted probabilities
                    model_output = model(enc_input, dec_input, src_mask, tgt_mask, None, None, None)
                else:
                    enc_input = enc_input.reshape(enc_input.shape[0], enc_input.shape[1]*enc_input.shape[2])
                    model_output = model(enc_input)
                    model_output = model_output.view(batch_size, prediction_len, -1)  # Reshape to [batch_size, prediction_len, num_classes]

                # Check if the predicted class (max index) matches the ground truth class
                for j in range(prediction_len):
                    predicted_probs = model_output[:, j, :]  # Predicted probability for each time step
                    predicted_classes = torch.argmax(predicted_probs, dim=-1)  # Get the index of the max probability
                    
                    correct_predictions_dict[model_name_list[i]] += torch.sum(predicted_classes == batch_classes[:, j]).item()
                    total_predictions += batch_size
        
    # Calculate the accuracy
    for model_name in model_name_list:
        correct_predictions = correct_predictions_dict[model_name]
        accuracy = correct_predictions / total_predictions
        print(f"Accuracy for {model_name}: {accuracy * 100:.2f}%")

def plot_predictions_for_sample(model, dataset, vocab_dict, prediction_len, sample_idx, device=DEVICE):
    """
    Plot the predicted probabilities for a random sample and mark the correct class index.
    
    Args:
    - model: Trained Seq2SeqWithEmbeddingmod model.
    - dataset: Test dataset (tensor).
    - vocab_dict: Dictionary mapping 44-size vectors to class indices.
    - prediction_len: Length of the prediction (number of tokens to predict).
    - sample_idx: The index of the sample to visualize.
    - device: Device to run the test on ('cuda' or 'cpu').
    """
    model.to(device)  # Move the model to the specified device
    model.eval()  # Set model to evaluation mode
    
    with torch.no_grad():
        # Get the sample input
        sample = dataset[sample_idx:sample_idx + 1, :, :].clone()
        enc_input = sample[:, :-prediction_len, :].to(device)
        dec_input = (1.5 * torch.ones((1, prediction_len, sample.shape[2]))).to(device)
        
        src_mask, tgt_mask, _, _ = create_mask(enc_input, dec_input, pad_idx=PAD_IDX, device=device)
        expected_output = sample[:, -prediction_len:, :].to(device)
        
        # Forward pass through the model to get the predicted probabilities
        model_output = model(enc_input, dec_input, src_mask, tgt_mask, None, None, None)
        
        # Convert expected_output to class indices (ground truth)
        correct_classes = []
        for t in range(prediction_len):
            vector = tuple(expected_output[0, t, :].tolist())
            class_idx = vocab_dict.get(vector, -1)  # Get class index from vocab_dict
            if class_idx == -1:
                raise ValueError(f"Vector not found in vocab_dict: {vector}")
            correct_classes.append(class_idx)
        
        # Plot the prediction graph
        i = random.randint(0, prediction_len - 1)  # Randomly select a time step
        # for i in range(prediction_len):
        predicted_probs = model_output[:, i, :].squeeze().cpu().numpy()  # Get predicted probabilities for each step
            
        # Plot the predicted probabilities for the current time step
        plt.figure(figsize=(10, 6))
        plt.plot(predicted_probs, label='Predicted Probabilities')
        plt.axvline(x=correct_classes[i], color='r', linestyle='--', label='Correct Class Index')
        plt.title(f'Prediction at time step {i+1} (Sample index: {sample_idx})')
        plt.xlabel('Class Index')
        plt.ylabel('Probability')
        plt.legend()
        plt.grid(True)
        plt.savefig(str(ITER)+'iter-sample'+str(sample_idx)+'-pred'+str(i)+'.png')

def plot_class_distribution(true_classes, predicted_classes, vocab_size, iteration):
    """
    Plot the distribution of true classes and predicted classes.
    
    Args:
    - true_classes: List of true class indices.
    - predicted_classes: List of predicted class indices.
    - vocab_size: The number of unique classes (size of the vocabulary).
    - iteration: The iteration number for saving the figure.
    """
    # Count the occurrences of each class in the true and predicted classes
    true_class_counts = [0] * vocab_size
    predicted_class_counts = [0] * vocab_size
    
    for true_class in true_classes:
        true_class_counts[true_class] += 1
    
    for pred_class in predicted_classes:
        predicted_class_counts[pred_class] += 1
    
    total_true = len(true_classes)
    total_predicted = len(predicted_classes)
    
    true_class_percentage = [(count / total_true) * 100 for count in true_class_counts]
    predicted_class_percentage = [(count / total_predicted) * 100 for count in predicted_class_counts]
    
    # Plot the class distributions as line plots
    plt.figure(figsize=(12, 6))
    
    plt.plot(range(vocab_size), true_class_percentage, label='True Class Distribution (%)', color='b', marker='o')
    plt.plot(range(vocab_size), predicted_class_percentage, label='Predicted Class Distribution (%)', color='r', marker='x')
    
    plt.xlabel('Class Index')
    plt.ylabel('Percentage (%)')
    plt.title('True vs Predicted Class Distribution (Percentage)')
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    plt.savefig(f'{iteration}_class_distribution.png')
    plt.show()

def test_and_plot_distribution(model, dataset, vocab_dict, prediction_len, batch_size=32, vocab_size=3231, iteration=""):
    """
    Test model and plot class distributions for true and predicted labels.
    
    Args:
    - model: Trained model for testing.
    - dataset: Dataset to test on.
    - vocab_dict: Dictionary mapping 44-size vectors to class indices.
    - prediction_len: Length of the prediction.
    - batch_size: Batch size.
    - vocab_size: Number of classes in the vocabulary.
    - iteration: Iteration number for saving files.
    """
    model.eval()  # Set model to evaluation mode
    true_classes_list = []
    predicted_classes_list = []
    
    num_batches = dataset.shape[0] // batch_size
    with torch.no_grad():
        for batch in range(num_batches):
            input_batch = dataset[batch * batch_size:(batch + 1) * batch_size, :, :].clone()
            enc_input = input_batch[:, :-prediction_len, :].to(DEVICE)
            dec_input = (1.5 * torch.ones((batch_size, prediction_len, input_batch.shape[2]))).to(DEVICE)
            
            src_mask, tgt_mask, _, _ = create_mask(enc_input, dec_input, pad_idx=PAD_IDX, device=DEVICE)
            expected_output = input_batch[:, -prediction_len:, :].to(DEVICE)

            model_output = model(enc_input, dec_input, src_mask, tgt_mask, None, None, None)
            
            batch_true_classes = []
            
            # Convert expected_output to class indices (ground truth)
            for i in range(batch_size):
                for t in range(prediction_len):
                    vector = tuple(expected_output[i, t, :].tolist())
                    class_idx = vocab_dict.get(vector, -1)
                    if class_idx == -1:
                        raise ValueError(f"Vector not found in vocab_dict: {vector}")
                    batch_true_classes.append(class_idx)

            # Convert to tensor and move to device
            batch_true_classes = torch.tensor(batch_true_classes, dtype=torch.long).to(DEVICE)
            batch_true_classes = batch_true_classes.view(batch_size, prediction_len)

            for i in range(prediction_len):
                predicted_probs = model_output[:, i, :]  # Predicted probability for each time step
                predicted_classes = torch.argmax(predicted_probs, dim=-1)  # Get the index of the max probability
                
                # Collect true and predicted classes
                true_classes_list.extend(batch_true_classes[:, i].tolist())
                predicted_classes_list.extend(predicted_classes.tolist())

    # Plot the class distribution
    # plot_class_distribution(true_classes_list, predicted_classes_list, vocab_size, iteration)
    with open('NEWDatasets/combined-dataset-preprocessed/6col-VocabBackDict.p', 'rb') as f_vocab_back:
        vocab_back_dict = pickle.load(f_vocab_back)
    process_and_calculate_distances(np.array([vocab_back_dict[class_idx] for class_idx in true_classes_list]), np.array([vocab_back_dict[class_idx] for class_idx in predicted_classes_list]), bucket_boundaries_1x)

def compute_l1_distances(true_vectors, predicted_vectors):
    """
    Compute the distance for each sample as the sum of absolute differences 
    between true and predicted indices across the 5 features.

    Parameters
    ----------
    true_vectors : ndarray of shape (N, 5)
    predicted_vectors : ndarray of shape (N, 5)

    Returns
    -------
    distances : ndarray of shape (N,)
        distances[i] = sum of abs differences for sample i
    """
    # Assuming both arrays have the same shape: (N, 5)
    distances = np.sum(np.abs(true_vectors - predicted_vectors), axis=1)
    return distances

# Function to calculate bucket distances for a batch
def calculate_bucket_distances(true_values, predicted_values, bucket_boundaries):
    total_distance = []
    
    # Create a sorted list of the keys in bucket_boundaries
    sorted_keys = sorted(bucket_boundaries.keys())
    num_features = len(sorted_keys)
    print("num_features: ", num_features)
    
    # Create a list to hold distances for each feature
    feature_distances = [[] for _ in range(num_features)]
    
    # For each sample (vector)
    for vector_idx in range(true_values.shape[0]):
        vector_distance = 0
        current_idx = 0  # This will track the starting index of the next feature's one-hot encoding
        
        # For each feature using sorted keys
        for feature_idx in range(num_features):
            key = sorted_keys[feature_idx]  # Get the sorted key corresponding to the feature index
            
            # Get the bucket boundaries for this feature
            feature_buckets = bucket_boundaries[key]
            num_buckets = len(feature_buckets) + 1  # Number of buckets for this feature
            
            # Extract the one-hot encoded true and predicted values for this feature
            true_one_hot = true_values[vector_idx, current_idx:current_idx + num_buckets]
            predicted_one_hot = predicted_values[vector_idx, current_idx:current_idx + num_buckets]
            
            # Get the index of the bucket for true and predicted values
            true_bucket_idx = np.argmax(true_one_hot)  # Finds the index where the one-hot encoding is 1
            predicted_bucket_idx = np.argmax(predicted_one_hot)  # Same for predicted
            
            # Calculate the index difference (absolute distance)
            distance = abs(true_bucket_idx - predicted_bucket_idx)
            vector_distance += distance
            
            # Store the distance for the current feature
            feature_distances[feature_idx].append(distance)
            
            # Move the index to the next feature's one-hot encoding in the vector
            current_idx += num_buckets
        
        # Append the vector distance to the total list
        total_distance.append(vector_distance)

    return total_distance, feature_distances  # Return distances for each feature


def process_and_calculate_distances(true_vectors, predicted_vectors, bucket_boundaries):
    total_distances, feature_distances = calculate_bucket_distances(true_vectors, predicted_vectors, bucket_boundaries)
    
    # Plot distribution for each feature
    for feature_idx, distances in enumerate(feature_distances):
        unique_distances, counts = np.unique(distances, return_counts=True)
        
        # Convert counts to percentages
        distribution = counts / counts.sum() * 100
        
        # Plot the distribution
        plt.figure(figsize=(10, 6))
        plt.bar(unique_distances, distribution, color='blue')
        plt.xlabel('Distance from True Bucket')
        plt.ylabel('Percentage')
        plt.title(f'Distribution of Bucket Distances for Feature {feature_idx}')
        plt.grid(True)
        
        # Save each figure
        plt.savefig(f'{ITER}iter-bucket-distance-feature-{feature_idx}.png')
        plt.close()  # Close the figure to save memory

    # Optionally, you can also plot the overall distribution
    unique_total_distances, total_counts = np.unique(total_distances, return_counts=True)
    total_distribution = total_counts / total_counts.sum() * 100
    
    plt.figure(figsize=(10, 6))
    plt.bar(unique_total_distances, total_distribution, color='blue')
    plt.xlabel('Distance from True Bucket')
    plt.ylabel('Percentage')
    plt.title('Overall Distribution of Bucket Distances')
    plt.grid(True)
    plt.savefig(f'{ITER}iter-bucket-distance-total.png')
    plt.close()  # Close the figure to save memory
    
def test_and_plot_distribution_multi_model(model_list, model_name_list, is_transformer_list, datasets, vocab_dict, prediction_len, batch_size=32, vocab_size=3231, iteration=""):
    """
    Test multiple models and plot class distributions for true and predicted labels.
    
    Args:
    - model_list: List of trained models for testing.
    - model_name_list: List of model names for labeling.
    - is_transformer_list: List of booleans indicating whether the model is a transformer.
    - dataset: Dataset to test on.
    - vocab_dict: Dictionary mapping 44-size vectors to class indices.
    - prediction_len: Length of the prediction.
    - batch_size: Batch size.
    - vocab_size: Number of classes in the vocabulary.
    - iteration: Iteration number for saving files.
    """
    model_results = {model_name: {'true_classes': [], 'predicted_classes': []} for model_name in model_name_list}

    num_batches = datasets[0].shape[0] // batch_size
    with torch.no_grad():
        for i, model in enumerate(model_list):
            dataset = torch.from_numpy(datasets[i])
            for batch in range(num_batches):
                input_batch = dataset[batch * batch_size:(batch + 1) * batch_size, :, :].clone()
                enc_input = input_batch[:, :-prediction_len, :].to(DEVICE)
                dec_input = (1.5 * torch.ones((batch_size, prediction_len, input_batch.shape[2]))).to(DEVICE)
                
                src_mask, tgt_mask, _, _ = create_mask(enc_input, dec_input, pad_idx=PAD_IDX, device=DEVICE)
                expected_output = input_batch[:, -prediction_len:, :].to(DEVICE)

                batch_true_classes = []
                # Convert expected_output to class indices (ground truth)
                for j in range(batch_size):
                    for t in range(prediction_len):
                        vector = tuple(expected_output[j, t, 1:].tolist())
                        class_idx = vocab_dict.get(vector, -1)
                        if class_idx == -1:
                            raise ValueError(f"Vector not found in vocab_dict: {vector}")
                        batch_true_classes.append(class_idx)

                # Convert to tensor and move to device
                batch_true_classes = torch.tensor(batch_true_classes, dtype=torch.long).to(DEVICE)
                batch_true_classes = batch_true_classes.view(batch_size, prediction_len)

                # valid_mask = (expected_output[:, :, :].sum(dim=-1) != -1).to(DEVICE)

                # print("i:", i)
                # print("model_list: ", model_name_list)
                model_name = model_name_list[i]

                if is_transformer_list[i]:
                    # Forward pass through the transformer model to get the predicted probabilities
                    model_output = model(enc_input, dec_input, src_mask, tgt_mask, None, None, None)
                else:
                    # Forward pass through the MLP model
                    enc_input_flat = enc_input.reshape(enc_input.shape[0], enc_input.shape[1] * enc_input.shape[2])
                    model_output = model(enc_input_flat)
                    model_output = model_output.view(batch_size, prediction_len, -1)  # Reshape to [batch_size, prediction_len, num_classes]

                # Collect true and predicted classes
                for t in range(prediction_len):
                    predicted_probs = model_output[:, t, :]  # Predicted probability for each time step
                    predicted_classes = torch.argmax(predicted_probs, dim=-1)  # Get the index of the max probability
                    
                    model_results[model_name]['true_classes'].extend(batch_true_classes[:, t].tolist())
                    model_results[model_name]['predicted_classes'].extend(predicted_classes.tolist())

    with open(str(ITER)+'iter-model-results.p', 'wb') as f_results:
        pickle.dump(model_results, f_results, pickle.HIGHEST_PROTOCOL)
    
    # with open(str(ITER)+'iter-model-results.p', 'rb') as f_results:
    #     model_results = pickle.load(f_results)
    
    # Plot the distributions for both models
    with open('NEWDatasets/combined-dataset-preprocessed/6col-VocabBackDict.p', 'rb') as f_vocab_back:
        vocab_back_dict = pickle.load(f_vocab_back)

    plt.figure(figsize=(10, 6))
    cdf_dict = {}
    
    # Plot PDF and CDF for each model
    for model_name in model_name_list:
        print(f"Processing bucket distances for {model_name}...")
        true_classes = model_results[model_name]['true_classes']
        predicted_classes = model_results[model_name]['predicted_classes']
        
        true_vectors = np.array([vocab_back_dict[class_idx] for class_idx in true_classes])
        predicted_vectors = np.array([vocab_back_dict[class_idx] for class_idx in predicted_classes])

        # Calculate distances
        print("true: ", true_vectors)
        print("predicted: ", predicted_vectors)
        # total_distances, feature_distances = calculate_bucket_distances(true_vectors, predicted_vectors, bucket_boundaries_ccbench)
        total_distances = compute_l1_distances(true_vectors, predicted_vectors)

        # Plot the overall distribution for each model (PDF)
        unique_total_distances, total_counts = np.unique(total_distances, return_counts=True)
        total_distribution = total_counts / total_counts.sum() * 100
        # Cumulative sum to get CDF
        cdf = np.cumsum(total_counts / total_counts.sum() * 100)
        cdf_dict[model_name] = (unique_total_distances, cdf)
        
        # PDF Plot
        plt.plot(unique_total_distances, total_distribution, label=model_name)  # Line plot for PDF
        print("Total distribution (PDF):", total_distribution)

    plt.xlabel('Distance from True Bucket')
    plt.ylabel('Percentage')
    plt.title(f'PDF of Bucket Distances (Multiple Models)')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig(f'{iteration}_iter_bucket_distance_comparison_PDF.png')
    plt.show()

    # CDF Plot
    plt.figure(figsize=(10, 6))
    
    for model_name in model_name_list:
        
        unique_total_distances, cdf = cdf_dict[model_name]
        # CDF Plot
        plt.plot(unique_total_distances, cdf, label=model_name)
        print(f"Total CDF for {model_name}: {cdf}")

    plt.xlabel('Distance from True Bucket')
    plt.ylabel('Cumulative Percentage')
    plt.title(f'CDF of Bucket Distances (Multiple Models)')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig(f'{iteration}_iter_bucket_distance_comparison_CDF.png')
    plt.show()

def process_and_calculate_distances_multiple_models(bucket_boundaries):
    """
    Calculate and plot bucket distances for multiple models.

    Args:
    - true_vectors: The true vectors (ground truth).
    - predicted_vectors_dict: A dictionary where each key is a model name, and the value is the predicted vectors by that model.
    - bucket_boundaries: Dictionary of bucket boundaries for each feature.
    - model_names: List of model names for plotting.
    """
    # Dictionary to store total and feature distances for each model
    model_distances = {}
    with open('./NEWDatasets/'+str(ITER)+'iter-model-results.p', 'rb') as f_results:
        model_results = pickle.load(f_results)
    
    # Plot the distributions for both models
    with open('NEWDatasets/FullDataset1x-filtered1-bucketized-VocabBackDict.p', 'rb') as f_vocab_back:
        vocab_back_dict = pickle.load(f_vocab_back)

    plt.figure(figsize=(10, 6))
    for model_name in model_results.keys():
        print(f"Processing bucket distances for {model_name}...")
        true_classes = model_results[model_name]['true_classes']
        predicted_classes = model_results[model_name]['predicted_classes']
        true_vectors = np.array([vocab_back_dict[class_idx] for class_idx in true_classes])
        predicted_vectors = np.array([vocab_back_dict[class_idx] for class_idx in predicted_classes])

        # Calculate distances
        # total_distances, feature_distances = calculate_bucket_distances(true_vectors, predicted_vectors, bucket_boundaries_1x)
        total_distances = compute_l1_distances(true_vectors, predicted_vectors)
        # model_distances[model_name] = {'total': total_distances , 'features': feature_distances}
    
    # Plot feature-wise bucket distance distribution for each model
    num_features = len(bucket_boundaries)
    
    for feature_idx in range(num_features):
        plt.figure(figsize=(10, 6))
        for model_name in model_distances.keys():
            distances = model_distances[model_name]['features'][feature_idx]
            unique_distances, counts = np.unique(distances, return_counts=True)
            distribution = counts / counts.sum() * 100
            
            # Plot distribution for this model
            plt.bar(unique_distances, distribution, alpha=0.6, label=model_name)  # Use alpha for transparency
            
            zero_distance_percentage = distribution[unique_distances == 0].sum()
            print(f"  {model_name} feature {feature_idx} accuracy: {zero_distance_percentage:.2f}% ")
        
        plt.xlabel('Distance from True Bucket')
        plt.ylabel('Percentage')
        plt.title(f'Distribution of Bucket Distances for Feature {feature_idx}')
        plt.grid(True)
        plt.legend(loc='upper right')  # Add legend for model names
        plt.savefig(f'{ITER}iter-bucket-distance-feature-{feature_idx}.png')
        plt.close()  # Close the figure to save memory


# Test the model
with open('/u/janechen/Documents/Transformer-training/NEWDatasets/combined-dataset-preprocessed/6col-VocabDict.p', 'rb') as f_vocab:
    vocab_dict = pickle.load(f_vocab)

transformer_model_small_name = 'BaseTransformer3_64_5_5_16_4_lr_1e-05_vocab-'+str(ITER)+'iter'
transformer_model_small = torch.load('/datastor1/janec/complete-models/Checkpoint-Combined_10RTT_6col_Transformer3_64_5_5_16_4_lr_1e-05-999iter.p', map_location=DEVICE)
transformer_model_large_name = 'BaseTransformer3_256_10_10_256_8_lr_1e-05_vocab-'+str(ITER)+'iter'
transformer_model_large = torch.load('/datastor1/janec/complete-models/Checkpoint-Large_Combined_10RTT_6col_Transformer3_256_8_8_64_8_lr_1e-05-999iter.p', map_location=DEVICE)
# mlp_model_name = 'MLP-MS-Checkpoint-102-'+str(ITER)+'iter'
# mlp_model = torch.load('Models/'+mlp_model_name+'.p', map_location=DEVICE)
# rtt_model_name = 'RTT-Transformer_64_5_5_16_4_lr_1e-05_999iter'
# rtt_model = torch.load('/datastor1/janec/combined-dataset-model/Checkpoint-BaseTransformer3_64_5_5_16_4_lr_1e-05_vocab-999iter.p', map_location=DEVICE)
# time_model_name = 'Time-Transformer_64_5_5_16_4_lr_1e-05_'+str(ITER)+'iter'
# time_model = torch.load('Models/Time-Checkpoint-BaseTransformer3_64_5_5_16_4_lr_1e-05_vocab-'+str(ITER)+'iter.p', map_location=DEVICE)
# model_list = [rtt_model]#, time_model]
# model_name_list = [rtt_model_name]#, time_model_name]
model_list = [transformer_model_small, transformer_model_large]#, mlp_model]
model_name_list = [transformer_model_small_name, transformer_model_large_name]#, mlp_model_name]
is_transformer_list = [True, True]
with open('NEWDatasets/combined-dataset-preprocessed/6col-rtt-based-test.p', 'rb') as f:
    rtt_test_dataset = pickle.load(f)
prediction_len = 10
test_and_plot_distribution_multi_model(model_list, model_name_list, is_transformer_list, [rtt_test_dataset, rtt_test_dataset], vocab_dict, prediction_len, batch_size=32, vocab_size=3231, iteration=ITER)


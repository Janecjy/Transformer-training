import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from models import create_mask
import time
import pickle
import copy

#CONSTANTS
PAD_IDX = 2
BATCH_SIZE = 1024
NUM_EPOCHS = 250
CONTEXT_LENGTH = 32
PREDICTION_LENGTH = 32

torch.manual_seed(0)

def form_dataset_mod(filelist, context_len, prediction_len, input_dim=13):
    seq_len = context_len + prediction_len
    train_dataset = torch.zeros((1, input_dim, 500))
    # print('Started Forming Raw Dataset')
    files_per_thread = len(filelist)//10
    global_max = -10*np.ones(input_dim)
    for thread in range(10):
        print(f'Chunk {thread+1} of 10')
        d1 = torch.zeros((1, input_dim, 500))
        max_vals = -10*np.ones(input_dim)
        for file in filelist[thread*files_per_thread:(thread+1)*files_per_thread]:
            d = pd.read_table(file, delimiter=',', header=0, engine='python')
            d = d.replace(' -nan', 1.0)
            d = d.to_numpy(float)
            temp = [i if i != 0 else 1 for i in np.max(d, axis=0)]
            max_vals = np.maximum(temp, max_vals)
            d = torch.FloatTensor(d).T
            d = d.reshape(1,input_dim,500)
            d1 = torch.cat((d1,d), dim=0)
        train_dataset = torch.cat((train_dataset, d1[1:,:,:]), dim=0)
        global_max = np.maximum(global_max,max_vals)
    global_max = global_max[:, np.newaxis]
    global_max = np.repeat(global_max, 500, axis=1)
    global_max = global_max[np.newaxis, :, :]
    global_max = np.repeat(global_max, train_dataset.shape[0], axis=0)
    global_max = torch.FloatTensor(global_max)
    train_dataset = torch.div(train_dataset, global_max)
    print('Finished gathering data. Reshaping...')
    train_dataset = train_dataset[1:,:,:]
    num_splits = 500//seq_len
    mod_data = torch.zeros(1, input_dim, seq_len)
    for i in range(num_splits):
        mod_data = torch.cat((mod_data, train_dataset[:,:, i*seq_len:(i+1)*seq_len]), axis=0)
    mod_data = mod_data[1:,:,:]
    mod_data = torch.transpose(mod_data, 1, 2)
    return mod_data, global_max[0,:,0]

def form_dataset(filelist, context_len, prediction_len, input_dim=13):
    seq_len = context_len + prediction_len
    train_dataset = torch.zeros((1, input_dim, 500))
    # print('Started Forming Raw Dataset')
    files_per_thread = len(filelist)//10
    for thread in range(10):
        print(f'Chunk {thread+1} of 10')
        d1 = torch.zeros((1, input_dim, 500))
        for file in filelist[thread*files_per_thread:(thread+1)*files_per_thread]:
            d = pd.read_table(file, delimiter=',', header=0, engine='python')
            d = d.replace(' -nan', 1.0)
            d = d.to_numpy(float)
            d = d/[i if i != 0 else 1 for i in np.max(d, axis=0)]
            d = torch.FloatTensor(d).T
            d = d.reshape(1,input_dim,500)
            d1 = torch.cat((d1,d), dim=0)
        train_dataset = torch.cat((train_dataset, d1[1:,:,:]), dim=0)
    print('Finished gathering data. Reshaping...')
    train_dataset = train_dataset[1:,:,:]
    num_splits = 500//seq_len
    mod_data = torch.zeros(1, input_dim, seq_len)
    for i in range(num_splits):
        mod_data = torch.cat((mod_data, train_dataset[:,:, i*seq_len:(i+1)*seq_len]), axis=0)
    mod_data = mod_data[1:,:,:]
    mod_data = torch.transpose(mod_data, 1, 2)
    return mod_data
    
def train_model(model, dataset, optimizer, prediction_len, device, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, checkpoint_suffix=None):
    loss_func = torch.nn.MSELoss(reduction='sum')
    loss_traj = []
    model.train()
    num_batch = dataset.shape[0]//batch_size
    for epoch in range(num_epochs):
        
        epoch_loss = 0.0
        t0 = time.time()
        for batch in range(num_batch):
            input = dataset[batch*batch_size:(batch+1)*batch_size, :, :].clone()
            enc_input = input[:, :-prediction_len, :].to(device)
            dec_input = (1.5*torch.ones((batch_size, prediction_len, input.shape[2]))).to(device)
            src_mask, tgt_mask, _, _ = create_mask(enc_input, dec_input, pad_idx=PAD_IDX, device=device)
            expected_output = input[:, -prediction_len:, :].to(device)
            model_out = model(enc_input, dec_input, src_mask, tgt_mask, None, None, None)
            optimizer.zero_grad()
            expected_shape = model_out.shape[-2]*model_out.shape[-1]
            loss = loss_func(model_out.reshape(-1, expected_shape), expected_output.reshape(-1, expected_shape))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_time = time.time() - t0
        epoch_loss /= num_batch
        loss_traj += [epoch_loss]
        
        print(f"[info] epoch {epoch} | Time taken = {epoch_time:.1f} seconds")
        print(f"Epoch loss = {epoch_loss:.6f}")
        if (epoch+1)%10 == 0:
            print(f"Epoch loss = {epoch_loss:.6f}")
        if epoch == num_epochs-1:
            print(f"Final Epoch: Loss = {epoch_loss:.6f}")
            if checkpoint_suffix is not None:
                with open('./Loss_log_'+checkpoint_suffix+'.p', 'wb') as f:
                    pickle.dump(loss_traj, f, protocol=pickle.HIGHEST_PROTOCOL)
                torch.save(model, './Models/'+checkpoint_suffix+'-1000iter.p')

        if epoch == 249 and checkpoint_suffix is not None:
            torch.save(model, './Models/Checkpoint-'+checkpoint_suffix+'-250iter.p')
        if epoch == 499 and checkpoint_suffix is not None:
            torch.save(model, './Models/Checkpoint-'+checkpoint_suffix+'-500iter.p')

        shuffle_idx = torch.randperm(dataset.shape[0])
        dataset = dataset[shuffle_idx, :, :]
    return model, loss_traj


def train_model_long(model, dataset, optimizer, prediction_len, min_context_len, device, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, checkpoint_suffix=None):
    loss_func = torch.nn.MSELoss(reduction='sum')
    loss_traj = []
    model.train()
    num_batch = dataset.shape[0]//batch_size
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        t0 = time.time()
        hide = np.random.choice(np.arange(0, 100-min_context_len-1, 1), num_batch, replace=True)
        for batch in range(num_batch):
            input = dataset[batch*batch_size:(batch+1)*batch_size, :, :].clone()
            enc_input = input[:, :-prediction_len, :].to(device)
            enc_input[:, :hide[batch], :] = 2.0*torch.ones((batch_size, hide[batch], input.shape[2]))
            enc_input = enc_input.to(device)
            dec_input = (1.5*torch.ones((batch_size, prediction_len, input.shape[2]))).to(device)
            src_mask, tgt_mask, _, _ = create_mask(enc_input, dec_input, pad_idx=PAD_IDX, device=device)
            expected_output = input[:, -prediction_len:, :].to(device)
            model_out = model(enc_input, dec_input, src_mask, tgt_mask, None, None, None)
            optimizer.zero_grad()
            expected_shape = model_out.shape[-2]*model_out.shape[-1]
            loss = loss_func(model_out.reshape(-1, expected_shape), expected_output.reshape(-1, expected_shape))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_time = time.time() - t0
        epoch_loss /= num_batch
        loss_traj += [epoch_loss]
        
        print(f"[info] epoch {epoch} | Time taken = {epoch_time:.1f} seconds")
        if (epoch+1)%10 == 0:
            print(f"Epoch loss = {epoch_loss:.6f}")
        if epoch == num_epochs-1:
            print(f"Final Epoch: Loss = {epoch_loss:.6f}")
            if checkpoint_suffix is not None:
                with open('./Loss_log_'+checkpoint_suffix+'.p', 'wb') as f:
                    pickle.dump(loss_traj, f, protocol=pickle.HIGHEST_PROTOCOL)
                torch.save(model, './Models/'+checkpoint_suffix+'-1000iter.p')

        if epoch == 249 and checkpoint_suffix is not None:
            torch.save(model, './Models/Checkpoint-'+checkpoint_suffix+'-250iter.p')
        if epoch == 499 and checkpoint_suffix is not None:
            torch.save(model, './Models/Checkpoint-'+checkpoint_suffix+'-500iter.p')

        shuffle_idx = torch.randperm(dataset.shape[0])
        dataset = dataset[shuffle_idx, :, :]
    return model, loss_traj

class weighted_mse():
    def __init__(self, weights, device, dim):
        self.weights = weights[:, np.newaxis]
        self.weights = np.repeat(self.weights, dim, axis=1)
        self.weights = torch.FloatTensor(self.weights).to(device)

    def loss(self,input, target):
        mse = (input-target)**2
        mse = torch.sum(mse, 0)
        mse = mse*self.weights
        return torch.sum(mse)

class weighted_mse_original():
    def __init__(self, weights, device, dim):
        self.weights = weights[:, np.newaxis]
        self.weights = np.repeat(self.weights, dim, axis=1)
        self.weights = torch.FloatTensor(self.weights).to(device)

    def loss(self,input, target):
        mse = (input-target)**2
        mse = torch.sum(mse, 0)
        mse = mse*self.weights
        return torch.sum(mse)

class weighted_mse_scale_mse():
    def __init__(self, weights, device, dim, lw, normalizer, alpha, selected_indices):
        self.weights = weights[:, np.newaxis]
        self.weights = np.repeat(self.weights, dim, axis=1)
        self.weights = torch.FloatTensor(self.weights).to(device)
        self.loss_weights = lw
        self.normalizer = normalizer
        self.alpha = alpha
        self.selected_indices = selected_indices  # List of feature indices for which different checks are applied

    def loss(self, input, target):
        # print("input: ", input.shape)
        # print("target: ", target.shape)
        # exit(0)
        mse = (input - target) ** 2
        
        # Apply the scaling factor only to the selected dimensions before summing
        for i, scale in enumerate(self.loss_weights):
            mse[:, :, i] *= scale
        
        mse = torch.sum(mse, 0)  # Sum across the batch dimension (dim 0)
        mse = mse * self.weights  # Apply the weights
        
        correctness_term = self.check_characteristic_correctness(input, target)
        # print("mse: ", torch.sum(mse))
        # print("correctness_term: ", correctness_term)
        # exit(0)
        total_loss = torch.sum(mse) + self.alpha * (100-correctness_term)
        
        return total_loss
    
    # Function to check if the entire sequence is stable (fluctuates within ±2)
    def check_stable(self, tokens):
        max_val = torch.max(tokens)
        min_val = torch.min(tokens)
        
        # Check if the range of the values is within ±2
        return (max_val - min_val) < 4
    
    # Function to check for continuous patterns in tokens
    def check_continuous_pattern(self, tokens):
        increases, decreases = 0, 0
        
        # Loop over the token positions
        for i in range(len(tokens) - 2):
            # Check for a continuous increase
            if tokens[i] < tokens[i + 1] < tokens[i + 2]:
                increases += 1
            # Check for a continuous decrease
            elif tokens[i] > tokens[i + 1] > tokens[i + 2]:
                decreases += 1

        return increases, decreases

    # Function to check if the values fall into specific buckets (binary indicator)
    def check_buckets(self, tokens, buckets):
        bucket_indicators = {bucket: 0 for bucket in buckets}
        for value in tokens:
            for bucket in buckets:
                if bucket[0] <= value < bucket[1]:
                    bucket_indicators[bucket] = 1
                    break
        return bucket_indicators

    def check_characteristic_correctness(self, input, target):
        correctness = 0
        total_checks = 0
        
        # Define bucket ranges for different features
        feature_buckets = {
            4: [(20, 40), (40, 60), (60, 80), (80, np.inf)],  # Buckets for feature index 5
            8: [(0, 5), (5, 10), (10, 20), (20, 30), (30, np.inf)]  # Buckets for feature index 9
        }
        
        # Iterate over both feature dimensions
        for feature in range(input.shape[2]):
            for i in range(input.shape[0]):  # Iterate through samples
                input_seq = input[i, :, feature].detach() * self.normalizer[feature]  # Get feature sequence for each sample
                target_seq = target[i, :, feature].detach() * self.normalizer[feature]  # Get corresponding target sequence
                
                if self.selected_indices[feature] in feature_buckets:
                    buckets = feature_buckets[self.selected_indices[feature]]
                    input_buckets = self.check_buckets(input_seq.cpu().numpy(), buckets)
                    target_buckets = self.check_buckets(target_seq.cpu().numpy(), buckets)
                    # print("input_buckets: ", input_buckets)
                    # print("target_buckets: ", target_buckets)
                    
                    # Check if both input and target have the same bucket distribution
                    if input_buckets == target_buckets:
                        # print("Correct bucket distribution")
                        if self.selected_indices[feature] == 8:
                            # Also check continuous and stable characteristics
                            input_continuous = self.check_continuous_pattern(input_seq)
                            input_stable = self.check_stable(input_seq)

                            target_continuous = self.check_continuous_pattern(target_seq)
                            target_stable = self.check_stable(target_seq)
                            # print("input_continuous: ", input_continuous)
                            # print("input_stable: ", input_stable)
                            # print("target_continuous: ", target_continuous)
                            # print("target_stable: ", target_stable)

                            # Correctness: True if both input and target exhibit the same patterns
                            if (input_continuous == target_continuous) and (input_stable == target_stable):
                                # print("Correct continuous pattern and stability")
                                correctness += 1  # Reward for matching patterns
                            if (input_continuous != target_continuous):
                                print("Incorrect continuous pattern")
                            if (input_stable != target_stable):
                                print("Incorrect stability")
                        else:
                            correctness += 1  # Reward for matching bucket distributions
                    else:
                        print("Incorrect bucket distribution")
                    
                total_checks += 1  # Increment the number of checks

        return correctness / total_checks * 100  # Return percentage correctness
    
def train_model_reweighted0(model, dataset, optimizer, weights, prediction_len, device, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, checkpoint_suffix=None):  
    # model 1: 2*loss, model 2: 2x scale mse, model 3: 2x output and target
    # torch.manual_seed(200)
    torch.save(model, './Models/tmp.p')
    
    loss_func1 = weighted_mse_original(torch.FloatTensor(weights).detach().clone(), device, dataset.shape[2])
    loss_traj1 = []
    model1 = torch.load('./Models/tmp.p')
    model1.train()
    optimizer1 = optimizer[0]
    
    # loss_func2 = weighted_mse_scale_mse(torch.FloatTensor(weights).detach().clone(), device, dataset.shape[2])
    # loss_traj2 = []
    # model2 = torch.load('./Models/tmp.p')
    # model2.train()
    # optimizer2 = optimizer[1]
    
    # loss_func3 = weighted_mse_original(copy.deepcopy(weights), device, dataset.shape[2])
    # loss_traj3 = []
    # model3 = copy.deepcopy(model)
    # model3.train()
    # optimizer3 = copy.deepcopy(optimizer)
    
    num_batch = dataset.shape[0]//batch_size
    for epoch in range(num_epochs):
        
        epoch_loss1, epoch_loss2, epoch_loss3 = 0.0, 0.0, 0.0
        t0 = time.time()
        for batch in range(num_batch):
            input = dataset[batch*batch_size:(batch+1)*batch_size, :, :].clone()
            enc_input = input[:, :-prediction_len, :].to(device)
            dec_input = (1.5*torch.ones((batch_size, prediction_len, input.shape[2]))).to(device)
            src_mask, tgt_mask, _, _ = create_mask(enc_input, dec_input, pad_idx=PAD_IDX, device=device)
            expected_output = input[:, -prediction_len:, :].to(device)
            
            # model 1 training            
            # model_out1 = model1(enc_input.detach().clone(), dec_input.detach().clone(), src_mask.detach().clone(), tgt_mask.detach().clone(), None, None, None)
            model_out1 = model1(enc_input, dec_input, src_mask, tgt_mask, None, None, None)
            optimizer1.zero_grad()
            loss1 = 20*loss_func1.loss(model_out1, expected_output)
            loss1.backward()
            optimizer1.step()
            epoch_loss1 += loss1.item()
            
            # # model 2 training
            # model_out2 = model2(enc_input.detach().clone(), dec_input.detach().clone(), src_mask.detach().clone(), tgt_mask.detach().clone(), None, None, None)
            # optimizer2.zero_grad()
            # loss2 = loss_func2.loss(model_out2, expected_output)
            # loss2.backward()
            # optimizer2.step()
            # epoch_loss2 += loss2.item()
            
            # # model 3 training
            # model_out3 = model3(copy.deepcopy(enc_input), copy.deepcopy(dec_input), copy.deepcopy(src_mask), copy.deepcopy(tgt_mask), None, None, None)
            # optimizer3.zero_grad()
            # scaling_factor = torch.sqrt(torch.tensor(2.0))
            # model_out_scaled = model_out3.clone()
            # expected_output_scaled = expected_output.clone()
            # model_out_scaled[:, :, :] *= scaling_factor
            # expected_output_scaled[:, :, :] *= scaling_factor
            # loss3 = loss_func3.loss(model_out_scaled, expected_output_scaled)
            # loss3.backward()
            # optimizer3.step()
            # epoch_loss3 += loss3.item()

            print(f"Batch {batch+1}/{num_batch}: Loss1 = {loss1.item():.6f}", flush=True)#, Loss2 = {loss2.item():.6f}, Loss3 = {loss3.item():.6f}", flush=True)

        epoch_time = time.time() - t0
        epoch_loss1 /= num_batch
        loss_traj1 += [epoch_loss1]
        # epoch_loss2 /= num_batch
        # loss_traj2 += [epoch_loss2]
        # epoch_loss3 /= num_batch
        # loss_traj3 += [epoch_loss3]
        
        print(f"[info] epoch {epoch} | Time taken = {epoch_time:.1f} seconds")
        print(f"Epoch loss 1 = {epoch_loss1:.6f}", flush=True)
        # print(f"Epoch loss 2 = {epoch_loss2:.6f}", flush=True)
        # print(f"Epoch loss 3 = {epoch_loss3:.6f}", flush=True)
        # if (epoch+1)%10 == 0:
        #     print(f"Epoch loss 1 = {epoch_loss1:.6f}, Epoch loss 2 = {epoch_loss2:.6f}, Epoch loss 3 = {epoch_loss3:.6f}")
        if epoch == num_epochs-1:
            print(f"Final Epoch: Model 1 Loss = {epoch_loss1:.6f}, Model 2 Loss = {epoch_loss2:.6f}, Model 3 Loss = {epoch_loss3:.6f}")
            if checkpoint_suffix is not None:
                with open('./Loss1_log_'+checkpoint_suffix+'.p', 'wb') as f:
                    pickle.dump(loss_traj1, f, protocol=pickle.HIGHEST_PROTOCOL)
                with open('./Loss2_log_'+checkpoint_suffix+'.p', 'wb') as f:
                    pickle.dump(loss_traj2, f, protocol=pickle.HIGHEST_PROTOCOL)
                with open('./Loss3_log_'+checkpoint_suffix+'.p', 'wb') as f:
                    pickle.dump(loss_traj3, f, protocol=pickle.HIGHEST_PROTOCOL)
                torch.save(model1, './Models/'+checkpoint_suffix+'-1000iter-1.p')
                torch.save(model2, './Models/'+checkpoint_suffix+'-1000iter-2.p')
                torch.save(model3, './Models/'+checkpoint_suffix+'-1000iter-3.p')

        if epoch == 49 and checkpoint_suffix is not None:
            torch.save(model1, './Models/Checkpoint-'+checkpoint_suffix+'-50iter-1.p')
            torch.save(model2, './Models/Checkpoint-'+checkpoint_suffix+'-50iter-2.p')
            torch.save(model3, './Models/Checkpoint-'+checkpoint_suffix+'-50iter-3.p')
        if epoch == 99 and checkpoint_suffix is not None:
            torch.save(model1, './Models/Checkpoint-'+checkpoint_suffix+'-100iter-1.p')
            torch.save(model2, './Models/Checkpoint-'+checkpoint_suffix+'-100iter-2.p')
            torch.save(model3, './Models/Checkpoint-'+checkpoint_suffix+'-100iter-3.p')
        if epoch == 249 and checkpoint_suffix is not None:
            torch.save(model1, './Models/Checkpoint-'+checkpoint_suffix+'-250iter-1.p')
            torch.save(model2, './Models/Checkpoint-'+checkpoint_suffix+'-250iter-2.p')
            torch.save(model3, './Models/Checkpoint-'+checkpoint_suffix+'-250iter-3.p')
        if epoch == 499 and checkpoint_suffix is not None:
            torch.save(model1, './Models/Checkpoint-'+checkpoint_suffix+'-500iter-1.p')
            torch.save(model2, './Models/Checkpoint-'+checkpoint_suffix+'-500iter-2.p')
            torch.save(model3, './Models/Checkpoint-'+checkpoint_suffix+'-500iter-3.p')

        shuffle_idx = torch.randperm(dataset.shape[0])
        dataset = dataset[shuffle_idx, :, :]
    
def train_model_reweighted1(model, dataset, optimizer, weights, prediction_len, device, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, checkpoint_suffix=None):
    
    # torch.save(model, './Models/tmp.p')
    
    loss_func = weighted_mse(weights, device, dataset.shape[2])
    loss_traj = []
    # model = torch.load('./Models/tmp.p', map_location=device)
    model.train()
    num_batch = dataset.shape[0]//batch_size
    for epoch in range(num_epochs):
        
        epoch_loss = 0.0
        t0 = time.time()
        for batch in range(num_batch):
            input = dataset[batch*batch_size:(batch+1)*batch_size, :, :].clone()
            enc_input = input[:, :-prediction_len, :].to(device)
            dec_input = (1.5*torch.ones((batch_size, prediction_len, input.shape[2]))).to(device)
            src_mask, tgt_mask, _, _ = create_mask(enc_input, dec_input, pad_idx=PAD_IDX, device=device)
            expected_output = input[:, -prediction_len:, :].to(device)
            model_out = model(enc_input, dec_input, src_mask, tgt_mask, None, None, None)
            optimizer.zero_grad()
            expected_shape = model_out.shape[-2]*model_out.shape[-1]
            
            loss = 20.0*loss_func.loss(model_out, expected_output)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print(f"Batch {batch+1}/{num_batch}: Loss1 = {loss.item():.6f}", flush=True)
        epoch_time = time.time() - t0
        epoch_loss /= num_batch
        loss_traj += [epoch_loss]
        
        print(f"[info] epoch {epoch} | Time taken = {epoch_time:.1f} seconds, Loss = {epoch_loss:.6f}")
        if (epoch+1)%10 == 0:
            print(f"Epoch loss = {epoch_loss:.6f}")
        if epoch == num_epochs-1:
            print(f"Final Epoch: Loss = {epoch_loss:.6f}")
            if checkpoint_suffix is not None:
                with open('./Loss_log_'+checkpoint_suffix+'.p', 'wb') as f:
                    pickle.dump(loss_traj, f, protocol=pickle.HIGHEST_PROTOCOL)
                torch.save(model, './Models/'+checkpoint_suffix+'-1000iter.p')

        if epoch == 249 and checkpoint_suffix is not None:
            torch.save(model, './Models/Checkpoint-'+checkpoint_suffix+'-250iter.p')
        if epoch == 499 and checkpoint_suffix is not None:
            torch.save(model, './Models/Checkpoint-'+checkpoint_suffix+'-500iter.p')

        shuffle_idx = torch.randperm(dataset.shape[0])
        dataset = dataset[shuffle_idx, :, :]
        
def train_model_reweighted2(model, dataset, optimizer, weights, prediction_len, device, lw, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, checkpoint_suffix=None, normalizer=None, alpha=0, selected_indices=None):
    
    loss_func = weighted_mse_scale_mse(weights, device, dataset.shape[2], lw, normalizer, alpha, selected_indices)
    loss_traj = []
    model.train()
    num_batch = dataset.shape[0]//batch_size
    for epoch in range(num_epochs):
        
        epoch_loss = 0.0
        t0 = time.time()
        for batch in range(num_batch):
            input = dataset[batch*batch_size:(batch+1)*batch_size, :, :].clone()
            enc_input = input[:, :-prediction_len, :].to(device)
            dec_input = (1.5*torch.ones((batch_size, prediction_len, input.shape[2]))).to(device)
            src_mask, tgt_mask, _, _ = create_mask(enc_input, dec_input, pad_idx=PAD_IDX, device=device)
            expected_output = input[:, -prediction_len:, :].to(device)
            model_out = model(enc_input, dec_input, src_mask, tgt_mask, None, None, None)
            optimizer.zero_grad()
            expected_shape = model_out.shape[-2]*model_out.shape[-1]
            
            loss = loss_func.loss(model_out, expected_output)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            # print(f"Batch {batch+1}/{num_batch}: Loss2 = {loss.item():.6f}", flush=True)
        epoch_time = time.time() - t0
        epoch_loss /= num_batch
        loss_traj += [epoch_loss]
        
        print(f"[info] epoch {epoch} | Time taken = {epoch_time:.1f} seconds, Loss = {epoch_loss:.6f}")
        if (epoch+1)%10 == 0:
            print(f"Epoch loss = {epoch_loss:.6f}")
        if epoch == num_epochs-1:
            print(f"Final Epoch: Loss = {epoch_loss:.6f}")
            if checkpoint_suffix is not None:
                with open('./Loss_log_'+checkpoint_suffix+'.p', 'wb') as f:
                    pickle.dump(loss_traj, f, protocol=pickle.HIGHEST_PROTOCOL)
                torch.save(model, './Models/'+checkpoint_suffix+'-500iter.p')

        if epoch == 249 and checkpoint_suffix is not None:
            torch.save(model, './Models/Checkpoint-'+checkpoint_suffix+'-250iter.p')
        if epoch == 499 and checkpoint_suffix is not None:
            torch.save(model, './Models/Checkpoint-'+checkpoint_suffix+'-500iter.p')

        shuffle_idx = torch.randperm(dataset.shape[0])
        dataset = dataset[shuffle_idx, :, :]

def test_model(model, dataset, prediction_len, device):
    model = model.eval()
    loss_func = nn.MSELoss(reduction='sum')
    num_samples = dataset.shape[0]
    num_features = dataset.shape[2]
    test_loss = np.zeros((num_samples, prediction_len))
    outputs = []
    print(f'Total test samples = {test_loss.shape[0]}')
    for i in range(num_samples):
        sample = (dataset[i,:,:].reshape(1, dataset.shape[1], dataset.shape[2])).clone()
        enc_input = sample[:,:-prediction_len, :].to(device)
        dec_input = (1.5*torch.ones((1, prediction_len, sample.shape[2]))).to(device)
        expected_output = sample[:, -prediction_len:, :].to(device)
        src_mask, tgt_mask, _, _ = create_mask(enc_input, dec_input, pad_idx=PAD_IDX, device=device)
        model_out = model(enc_input, dec_input, src_mask, tgt_mask, None, None, None)
        # Calculate loss for each feature and each time step in the current batch
        for j in range(prediction_len):
            for k in range(num_features):
                test_loss[i, j, k] = loss_func(model_out[:, j, k], expected_output[:, j, k]).item()
    
    # Compute average test loss across all batches and prediction steps for each feature
    average_test_loss_per_feature = np.mean(test_loss, axis=(0, 1))
    # outputs = np.concatenate(outputs, axis=0)  # Convert list of arrays to a single NumPy array
    
    return average_test_loss_per_feature, outputs
        #     test_loss[i,:] = [loss_func(model_out[:,j,:], expected_output[:,j,:]).item() for j in range(prediction_len)]
    #     if i%(num_samples//10) == 0: print(f'Done testing {i} of {num_samples}')
    # return test_loss



def test_model_batched(model, dataset, batch_size, prediction_len, device, mae):
    model = model.eval()
    if not mae: loss_func = nn.MSELoss(reduction='mean')
    else: loss_func = nn.L1Loss(reduction='mean')
    num_samples = dataset.shape[0]
    num_features = dataset.shape[2]
    print(f'Total test samples = {num_samples}')
    num_batches = num_samples//batch_size
    test_loss = np.zeros((num_batches, prediction_len, num_features))
    outputs = []
    for i in range(num_batches):
        print(f'Starting batch {i+1} of {num_batches}')
        sample = dataset[i*batch_size:(i+1)*batch_size,:,:].clone()
        enc_input = sample[:,:-prediction_len, :].to(device)
        dec_input = (1.5*torch.ones((batch_size, prediction_len, sample.shape[2]))).to(device)
        expected_output = sample[:, -prediction_len:, :].to(device)
        src_mask, tgt_mask, _, _ = create_mask(enc_input, dec_input, pad_idx=PAD_IDX, device=device)
        model_out = model(enc_input, dec_input, src_mask, tgt_mask, None, None, None)
        # print("model_out shape: ", model_out.shape)
        # Calculate loss for each feature and each time step in the current batch
        for j in range(prediction_len):
            for k in range(num_features):
                test_loss[i, j, k] = loss_func(model_out[:, j, k], expected_output[:, j, k]).item()
    # print("test_loss shape: ", test_loss.shape)
    
    # Compute average test loss across all batches and prediction steps for each feature
    average_test_loss_per_feature = np.mean(test_loss, axis=0)
    # print("average_test_loss_per_feature shape: ", average_test_loss_per_feature.shape)
    # outputs = np.concatenate(outputs, axis=0)  # Convert list of arrays to a single NumPy array
    
    return average_test_loss_per_feature, outputs
    #     test_loss[i,:] = [loss_func(model_out[:,j,:], expected_output[:,j,:]).item() for j in range(prediction_len)]
    # test_loss = (1/batch_size)*test_loss
    # outputs = np.concatenate(outputs, axis=0)  # Convert list of arrays to a single NumPy array
    # return test_loss, outputs


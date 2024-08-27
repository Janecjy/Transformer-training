import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from models import create_mask
import time
import pickle

#CONSTANTS
PAD_IDX = 2
BATCH_SIZE = 1024
NUM_EPOCHS = 250
CONTEXT_LENGTH = 32
PREDICTION_LENGTH = 32

def form_dataset_mod(filelist, context_len, prediction_len, input_dim=13):
    seq_len = context_len + prediction_len
    train_dataset = torch.zeros((1, input_dim, 500))
    print('Started Forming Raw Dataset')
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
    print('Started Forming Raw Dataset')
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
    
def train_model_reweighted(model, dataset, optimizer, weights, prediction_len, device, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, checkpoint_suffix=None):
    loss_func = weighted_mse(weights, device, dataset.shape[2])
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
            
            # Apply the scaling factor only to the 6th column (index 5)
            # scaling_factor = torch.sqrt(torch.tensor(20.0))
            # # Separate the columns to be scaled and those not to be scaled
            # model_out_scaled = model_out.clone()
            # expected_output_scaled = expected_output.clone()
            # # print(model_out)

            # # Apply scaling factor to the 6th column (index 5)
            # model_out_scaled[:, :, :] *= scaling_factor
            # expected_output_scaled[:, :, :] *= scaling_factor
            # # print(model_out_scaled)

            # # Calculate loss using scaled outputs
            # loss = loss_func.loss(model_out_scaled, expected_output_scaled)
            # print(loss)
            # loss = loss_func(model_out.reshape(-1, expected_shape), expected_output.reshape(-1, expected_shape))
            loss = 20*loss_func.loss(model_out, expected_output)
            # print(loss2)
            # return
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


def test_model(model, dataset, prediction_len, device):
    model = model.eval()
    loss_func = nn.MSELoss(reduction='sum')
    num_samples = dataset.shape[0]
    test_loss = np.zeros((num_samples, prediction_len))
    print(f'Total test samples = {test_loss.shape[0]}')
    for i in range(num_samples):
        sample = (dataset[i,:,:].reshape(1, dataset.shape[1], dataset.shape[2])).clone()
        enc_input = sample[:,:-prediction_len, :].to(device)
        dec_input = (1.5*torch.ones((1, prediction_len, sample.shape[2]))).to(device)
        expected_output = sample[:, -prediction_len:, :].to(device)
        src_mask, tgt_mask, _, _ = create_mask(enc_input, dec_input, pad_idx=PAD_IDX, device=device)
        model_out = model(enc_input, dec_input, src_mask, tgt_mask, None, None, None)
        test_loss[i,:] = [loss_func(model_out[:,j,:], expected_output[:,j,:]).item() for j in range(prediction_len)]
        if i%(num_samples//10) == 0: print(f'Done testing {i} of {num_samples}')
    return test_loss



def test_model_batched(model, dataset, batch_size, prediction_len, device, mae):
    model = model.eval()
    if not mae: loss_func = nn.MSELoss(reduction='mean')
    else: loss_func = nn.L1Loss(reduction='mean')
    num_samples = dataset.shape[0]
    print(f'Total test samples = {num_samples}')
    num_batches = num_samples//batch_size
    test_loss = np.zeros((num_batches, prediction_len))
    outputs = []
    for i in range(num_batches):
        print(f'Starting batch {i+1} of {num_batches}')
        sample = dataset[i*batch_size:(i+1)*batch_size,:,:].clone()
        enc_input = sample[:,:-prediction_len, :].to(device)
        dec_input = (1.5*torch.ones((batch_size, prediction_len, sample.shape[2]))).to(device)
        expected_output = sample[:, -prediction_len:, :].to(device)
        src_mask, tgt_mask, _, _ = create_mask(enc_input, dec_input, pad_idx=PAD_IDX, device=device)
        model_out = model(enc_input, dec_input, src_mask, tgt_mask, None, None, None)
        outputs.append(model_out.cpu().detach().numpy())
        test_loss[i,:] = [loss_func(model_out[:,j,:], expected_output[:,j,:]).item() for j in range(prediction_len)]
    test_loss = (1/batch_size)*test_loss
    outputs = np.concatenate(outputs, axis=0)  # Convert list of arrays to a single NumPy array
    return test_loss, outputs


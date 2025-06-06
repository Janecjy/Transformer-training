import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import PathVariables
import torch
import torch.nn as nn
import pickle
import time
from utils import test_model_batched, test_model, weighted_mse

#CONSTANTS
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DEVICE = "cpu"
print(DEVICE)
PAD_IDX = 2
BATCH_SIZE = 1024
NUM_EPOCHS = 250
CONTEXT_LENGTH = 32
PREDICTION_LENGTH = 32

def train_linear_model(model, dataset, optimizer, prediction_len, device, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, checkpoint_suffix=None):
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
            expected_output = input[:, -prediction_len:, :].to(device)
            model_out = model(enc_input)
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
        shuffle_idx = torch.randperm(dataset.shape[0])
        dataset = dataset[shuffle_idx, :, :]
    return model, loss_traj


def train_mlp(model, dataset, optimizer, prediction_len, device, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, checkpoint_suffix=None):
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
            enc_input = enc_input.reshape(enc_input.shape[0], enc_input.shape[1]*enc_input.shape[2])
            expected_output = input[:, -prediction_len:, :].to(device)
            model_out = model(enc_input)
            optimizer.zero_grad()
            expected_shape = expected_output.shape[-2]*expected_output.shape[-1]
            loss = loss_func(model_out, expected_output.reshape(expected_output.shape[0], expected_shape))
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
        shuffle_idx = torch.randperm(dataset.shape[0])
        dataset = dataset[shuffle_idx, :, :]
    return model, loss_traj

def test_linear_model(model, dataset, prediction_len, device):
    model = model.eval()
    loss_func = nn.MSELoss(reduction='sum')
    num_samples = dataset.shape[0]
    print(f'Total test samples = {num_samples}')
    test_loss = np.zeros((num_samples, prediction_len))
    for i in range(num_samples):
        sample = (dataset[i, :,:].reshape(1, dataset.shape[-2], dataset.shape[-1])).clone()
        enc_input = sample[:, :-prediction_len, :].to(device)
        expected_output = sample[:, -prediction_len:, :].to(device)
        model_out = model(enc_input)
        test_loss[i,:] = [loss_func(model_out[:,j,:], expected_output[:,j,:]).item() for j in range(prediction_len)]
        if i%(num_samples//10) == 0: print(f'Done testing {i} of {num_samples}')
    return test_loss


def test_linear_model_batched(model, dataset, batch_size, prediction_len, device, mae=True):
    model = model.eval()
    if not mae:
        loss_func = nn.MSELoss(reduction='mean')
    else:
        loss_func = nn.L1Loss(reduction='mean')
    num_samples = dataset.shape[0]
    print(f'Total test samples = {num_samples}')
    num_batches = dataset.shape[0]//batch_size
    test_loss = np.zeros((num_batches, prediction_len))
    for i in range(num_batches):
        print(f'Starting Batch {i+1} of {num_batches}')
        sample = dataset[i*batch_size:(i+1)*batch_size, :,:].clone()
        enc_input = sample[:, :-prediction_len, :].to(device)
        expected_output = sample[:, -prediction_len:, :].to(device)
        model_out = model(enc_input)
        test_loss[i,:] = [loss_func(model_out[:,j,:], expected_output[:,j,:]).item() for j in range(prediction_len)]
    test_loss = (1/batch_size)*test_loss
    return test_loss


def test_mlp_batched(model, dataset, batch_size, prediction_len, device, mae=True):
    model = model.eval()
    if not mae:
        loss_func = nn.MSELoss(reduction='mean')
    else:
        loss_func = nn.L1Loss(reduction='mean')
    num_samples = dataset.shape[0]
    print(f'Total test samples = {num_samples}')
    num_batches = dataset.shape[0]//batch_size
    test_loss = np.zeros((num_batches, prediction_len))
    for i in range(num_batches):
        print(f'Starting Batch {i+1} of {num_batches}')
        sample = dataset[i*batch_size:(i+1)*batch_size, :,:].clone()
        enc_input = sample[:, :-prediction_len, :].to(device)
        enc_input = enc_input.reshape(enc_input.shape[0], enc_input.shape[1]*enc_input.shape[2])
        expected_output = sample[:, -prediction_len:, :].to(device)
        model_out = model(enc_input)
        test_loss[i,:] = [loss_func(model_out[:,j*13:(j+1)*13], expected_output[:,j,:]).item() for j in range(prediction_len)]
    test_loss = (1/batch_size)*test_loss
    return test_loss

def combined_loss_data(model_list, is_linear, test_dataset, pred_len, device, mae):
    test_loss = dict()
    for i in range(len(model_list)):
        model = torch.load('./Models/'+model_list[i]+'.p', map_location=device)
        print(model_list[i], sum(p.numel() for p in model.parameters() if p.requires_grad))
        if is_linear[i]:
            model_loss = test_linear_model(model, test_dataset, pred_len, device, mae)
        elif 'MLP' in model_list[i]:
            model_loss = test_mlp_batched(model, test_dataset, pred_len, device, mae)
        else:
            model_loss = test_model(model, test_dataset, pred_len, device, mae)
        mean = np.mean(model_loss, axis=0)
        test_loss[model_list[i]] = mean
    return test_loss


def combined_loss_data_batched(model_list, is_linear, test_dataset, batch_size, pred_len, device, mae):
    test_loss = dict()
    for i in range(len(model_list)):
        model = torch.load('./Models/'+model_list[i]+'.p', map_location=device)
        print(model_list[i], sum(p.numel() for p in model.parameters() if p.requires_grad))
        if is_linear[i]:
            model_loss = test_linear_model_batched(model, test_dataset, batch_size, pred_len, device, mae)
        elif 'MLP' in model_list[i]:
            model_loss = test_mlp_batched(model, test_dataset, batch_size, pred_len, device, mae)
        else:
            model_loss = test_model_batched(model, test_dataset, batch_size, pred_len, device, mae)
        mean = np.mean(model_loss, axis=0)
        test_loss[model_list[i]] = mean
    return test_loss


def combined_plot(model_list, is_linear, labels, title, save_name, test_dataset, batch_size, pred_len, device, mae):
    if batch_size is not None:
        loss_dict = combined_loss_data_batched(model_list, is_linear, test_dataset, batch_size, pred_len, device, mae)
    else:
        loss_dict = combined_loss_data(model_list, is_linear, test_dataset, pred_len, device, mae)
    plt.figure(figsize=(12,8))
    for i in range(len(model_list)):
        plt.plot(np.arange(1, len(loss_dict[model_list[i]])+1, 1), loss_dict[model_list[i]], linewidth=3.0, label = labels[i])
        print(model_list[i], np.mean(loss_dict[model_list[i]]))
    plt.xlabel('Future Tokens')
    if not mae: plt.ylabel('Mean MSE Loss')
    else: plt.ylabel('Mean MAE')
    plt.legend()
    plt.title(title)
    plt.grid()
    if save_name is not None: 
        with open('./Loss_dict-'+save_name+'.p', 'wb') as f:
            pickle.dump(loss_dict, f, pickle.HIGHEST_PROTOCOL)
        plt.savefig(save_name+'.png', bbox_inches='tight')
    plt.show()
    plt.close()
            

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLinear(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, seq_len, pred_len, individual=True, channels = 6):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # Decompsition Kernel Size
        kernel_size = 7
        self.decompsition = series_decomp(kernel_size)
        self.individual = individual
        self.channels = channels

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len,self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len,self.pred_len))

                # Use this two lines if you want to visualize the weights
                # self.Linear_Seasonal[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
                # self.Linear_Trend[i].weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)
            
            # Use this two lines if you want to visualize the weights
            # self.Linear_Seasonal.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
            # self.Linear_Trend.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0,2,1), trend_init.permute(0,2,1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0),seasonal_init.size(1),self.pred_len],dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0),trend_init.size(1),self.pred_len],dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:,i,:] = self.Linear_Seasonal[i](seasonal_init[:,i,:])
                trend_output[:,i,:] = self.Linear_Trend[i](trend_init[:,i,:])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        x = seasonal_output + trend_output
        return x.permute(0,2,1) # to [Batch, Output length, Channel]
    
class NLinear(nn.Module):
    """
    Normalization-Linear
    """
    def __init__(self, seq_len, pred_len, individual=True, channels=6):
        super(NLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = channels
        self.individual = individual
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len,self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seq_last = x[:,-1:,:].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0),self.pred_len,x.size(2)],dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:,:,i] = self.Linear[i](x[:,:,i])
            x = output
        else:
            x = self.Linear(x.permute(0,2,1)).permute(0,2,1)
        x = x + seq_last
        return x # [Batch, Output length, Channel]


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=512) -> None:
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        if hidden_dim is not None: 
            self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(hidden_dim, hidden_dim),
                                  nn.ReLU(),
                                  nn.Linear(hidden_dim, output_dim))
        else:
            self.mlp = nn.Sequential(nn.Linear(input_dim, output_dim), nn.ReLU())
    def forward(self, x):
        x = self.mlp(x)
        return x

INPUT_DIM = int(sys.argv[1])
COLUMN_INDEX = int(sys.argv[2])
selected_indices = [0, 1, 4, 6, 8, 12]
with open('./NEWDatasets/FullDataset-new-train.p', 'rb') as f:
    train_dataset = pickle.load(f)
    train_dataset = train_dataset[:, :, selected_indices]
    if INPUT_DIM == 1:
        train_dataset = train_dataset[:, :, COLUMN_INDEX].unsqueeze(-1)
model = MLP(input_dim=CONTEXT_LENGTH*INPUT_DIM, output_dim=CONTEXT_LENGTH*INPUT_DIM, hidden_dim=102).to(DEVICE)
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
train_dataset = train_dataset.to(DEVICE)

opt = torch.optim.Adam(model.parameters(), lr=1e-4)
trained_model, loss_traj = train_mlp(model, train_dataset, opt, PREDICTION_LENGTH, DEVICE, 1000, BATCH_SIZE)

if INPUT_DIM == 1:
    torch.save(trained_model, './Models/MLP-norm-408dim-noweighting-1000iter-'+str(COLUMN_INDEX)+'.p')
else:
    torch.save(trained_model, './Models/MLP-norm-408dim-noweighting-selected-0, 1, 4, 5, 8, 12-1000iter.p')
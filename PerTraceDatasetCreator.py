import numpy as np
import matplotlib.pyplot as plt
import os
import PathVariables
import torch
import pandas as pd
import pickle

#CONSTANTS
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)
PAD_IDX = 2
BATCH_SIZE = 1024
NUM_EPOCHS = 250
CONTEXT_LENGTH = 32
PREDICTION_LENGTH = 32

def get_tput(filename, total_time=10, interval=0.02):
    tput_arr = pd.read_table(filename, delimiter=',', header=None, engine='python')
    tput_arr = tput_arr.to_numpy(float)
    tput = np.zeros((int(total_time/interval), 2))
    tput_size = tput_arr.shape[0]
    tput[:,0] = interval*np.arange(tput.shape[0])
    curr_line, count_packets = 0, 0
    for i in range(tput.shape[0]):
        if curr_line == tput_size: break
        curr_time = tput[i,0]
        if tput_arr[curr_line, 0] > curr_time+interval:
            continue
        else:
            count_packets = 0
            while tput_arr[curr_line, 0] <= curr_time+interval:
                count_packets += 1
                curr_line += 1
                if curr_line == tput_size: break
            tput[i,1] = 1e-6*count_packets*1448*8/interval
    return tput

def form_dataset_mod(filelist, context_len, prediction_len, input_dim=13):
    seq_len = context_len + prediction_len
    train_dataset = np.zeros((1, input_dim, 500))
    print('Started Forming Raw Dataset')
    files_per_thread = len(filelist)//10
    global_max = -10*np.ones(input_dim)
    for thread in range(10):
        print(f'Chunk {thread+1} of 10')
        d1 = np.zeros((1, input_dim, 500))
        max_vals = -10*np.ones(input_dim)
        for file in filelist[thread*files_per_thread:(thread+1)*files_per_thread]:
            try:
                d = pd.read_table(file[:-5], delimiter=',', header=0, engine='python')
                d = d.replace(' -nan', -1.0)
                d = d.to_numpy(float)
                dd = get_tput(file)[:,1]
                dd = dd[:, np.newaxis]
                d = np.hstack((d, dd))
                temp = [i if i != 0 else 1 for i in np.max(d, axis=0)]
                max_vals = np.maximum(temp, max_vals)
                d = d.T
                d = d.reshape(1,input_dim,500)
                d1 = np.vstack((d1,d))
            except:
                print(file)
                continue
        train_dataset = np.vstack((train_dataset, d1[1:,:,:]))
        global_max = np.maximum(global_max,max_vals)
    global_max = global_max[:, np.newaxis]
    global_max = np.repeat(global_max, 500, axis=1)
    global_max = global_max[np.newaxis, :, :]
    global_max = np.repeat(global_max, train_dataset.shape[0], axis=0)
    # global_max = torch.FloatTensor(global_max)
    train_dataset = np.divide(train_dataset, global_max)
    print('Finished gathering data. Reshaping...')
    train_dataset = train_dataset[1:,:,:]
    num_splits = 500//seq_len
    mod_data = np.zeros((1, input_dim, seq_len))
    for i in range(num_splits):
        mod_data = np.vstack((mod_data, train_dataset[:,:, i*seq_len:(i+1)*seq_len]))
    mod_data = mod_data[1:,:,:]
    mod_data = torch.FloatTensor(mod_data)
    global_max = torch.FloatTensor(global_max[0,:,0])
    mod_data = torch.transpose(mod_data, 1, 2)
    return mod_data, global_max

# out-of-distribution 
params_list_alt = [('Reno', 1, 2, 'ferry'), ('Reno', 5, 3, 'metro'), ('Reno', 2, 2, 'bus'), ('Reno', 1.5, 2, 'train'), ('Cubic', 0.9, 0.4, 'bus'), ('Cubic', 0.7, 0.8, 'metro'), ('Cubic', 0.5, 0.4, 'car'), ('Cubic', 0.8, 0.8, 'tram')]
params_list = [('Reno', 1, 4, 'metro'), ('Reno', 5, 1.5, 'ferry'), ('Reno', 2, 4, 'car'), ('Reno', 1.5, 4, 'tram'), ('Cubic', 0.9, 0.8, 'car'), ('Cubic', 0.8, 0.4, 'train'), ('Cubic', 0.7, 0.4, 'bus'), ('Cubic', 0.5, 0.8, 'ferry')]

filelist = []

for pol, inc, dec, transport in params_list_alt:
    if pol.startswith("Reno"): 
        continue
    else:
        path = PathVariables.cubic_files
        path = os.path.join(path, 'Cubic-'+str(inc)+'-'+str(dec), transport)
    tput_list = [os.path.join(path, f) for f in os.listdir(path)[:6] if os.path.isfile(os.path.join(path, f)) and f[-4:]=='tput'] # can modify this to sample differently
    l = [i for i in tput_list if os.path.exists(i[:-5])]
    l = l[:10000]
    data, normalizer = form_dataset_mod(l, CONTEXT_LENGTH, PREDICTION_LENGTH)

    data_dict = dict()
    data_dict['data'] = data
    data_dict['normalizer'] = normalizer
    import pickle
    with open('./NEWDatasets/Dataset' + 'Cubic-'+str(inc)+'-'+str(dec) + '.p', 'wb') as f:
        pickle.dump(data_dict, f, pickle.HIGHEST_PROTOCOL)
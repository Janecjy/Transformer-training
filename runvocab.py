import torch
from models import Seq2SeqWithEmbeddingmodClass, Seq2SeqWithEmbeddingmodClassMoreEmbedding
from utils import train_model, train_model_vocab, train_model_vocab_autoreg#, train_model_reweighted2, train_model_reweighted3
import pickle
import argparse
import numpy as np

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Function to parse a comma-separated string into a list of integers
def parse_indices(indices_str):
    return [int(i) for i in indices_str.split(',')]

parser = argparse.ArgumentParser()
parser.add_argument('--Dataset', '-Data', help='Filename prefix for Training Dataset in the NewDatasets Folder', type=str, default='FullDataset')
parser.add_argument('--GPUNumber', '-GPU', help='Index of GPU to use', type=int, default=0)
# parser.add_argument('--ModelName', '-MName', help='Save name for model and dataset', type=str, default='BaseTransformer3_norm')
parser.add_argument('--DimFeedForward', '-DFF', help='Dimension of Feed Forward Layer', type=int, default=256)
parser.add_argument('--NumEncoderLayers', '-NEL', help='Number of Encoder Layers', type=int, default=10)
parser.add_argument('--NumDecoderLayers', '-NDL', help='Number of Decoder Layers', type=int, default=10)
parser.add_argument('--EmbSize', '-ES', help='Embedding Size', type=int, default=32)
parser.add_argument('--NHead', '-NH', help='Number of Heads', type=int, default=4)
parser.add_argument('--Weighted', '-W', help='Whether to use weighted loss', type=str2bool, default=True)
parser.add_argument('--LearningRate', '-LR', help='Learning Rate', type=float, default=1e-4)
parser.add_argument('--AdamBeta1', '-AB1', help='Beta1 for Adam Optimizer', type=float, default=0.9)
parser.add_argument('--AdamBeta2', '-AB2', help='Beta2 for Adam Optimizer', type=float, default=0.999)
parser.add_argument('--SelectedIndices', '-SI', help='Comma-separated list of indices to select', type=parse_indices, default="0,1,2,3,4,5,6,7,8,9,10,11,12")
parser.add_argument('--LossWeight', '-LW', help='Weight for the loss function', type=parse_indices, default="1,1,1,1,1,1,1,1,1,1,1,1,1")
parser.add_argument('--Alpha', '-A', help='Alpha for the reweighted loss function', type=float, default=0.0)
parser.add_argument('--MoreEmbedding', '-ME', help='Whether to use more embedding layers', type=str2bool, default=False)
parser.add_argument('--PredictionLength', '-PL', help='Length of prediction', type=int, default=32)
parser.add_argument('--ContextLength', '-CL', help='Length of context', type=int, default=32)
args = parser.parse_args()
dataset_name = args.Dataset 
gpu = args.GPUNumber
dim = args.DimFeedForward
num_encoder_layers = args.NumEncoderLayers
num_decoder_layers = args.NumDecoderLayers
emb_size = args.EmbSize
nhead = args.NHead
weighted = args.Weighted
learning_rate = args.LearningRate
adam_beta1 = args.AdamBeta1
adam_beta2 = args.AdamBeta2
# selected_indices = [0, 1, 4, 6, 8, 12]
selected_indices = args.SelectedIndices
loss_weight = args.LossWeight
alpha = args.Alpha
save_name = "BaseTransformer3_"+str(dim)+"_"+str(num_encoder_layers)+"_"+str(num_decoder_layers)+"_"+str(emb_size)+"_"+str(nhead)+"_lr_"+str(learning_rate)+"_vocab"
more_embedding = args.MoreEmbedding

#CONSTANTS
DEVICE = torch.device("cuda:"+str(gpu) if torch.cuda.is_available() else "cpu")
import gc
torch.cuda.empty_cache()
gc.collect()
# DEVICE = torch.device("cpu")
print(DEVICE)
PAD_IDX = 2
BATCH_SIZE = 1024 * 2
NUM_EPOCHS = 1000
CONTEXT_LENGTH = args.ContextLength
PREDICTION_LENGTH = args.PredictionLength

with open('./NEWDatasets/'+dataset_name+'-train.p', 'rb') as f:
    train_dataset = pickle.load(f)
    # train_dataset = train_dataset[:, :, selected_indices]
    print(train_dataset.shape)

# with open('NEWDatasets/FullDataset.p', 'rb') as f:
#     d = pickle.load(f)
# N = d['normalizer'].detach().cpu().numpy()[selected_indices]

with open('NEWDatasets/FullDataset_alt-bucketized-VocabDict.p', 'rb') as f_vocab:
        vocab_dict = pickle.load(f_vocab)
        num_classes = len(vocab_dict)
        print("vocab dict size: ", num_classes)
    
if more_embedding:
    model = Seq2SeqWithEmbeddingmodClassMoreEmbedding(num_encoder_layers=num_encoder_layers,
                             num_decoder_layers=num_decoder_layers,
                             input_size=train_dataset.shape[-1],
                             emb_size=emb_size,
                             nhead=nhead,
                             dim_feedforward=dim,
                             dropout=0,
                             num_classes=num_classes).to(DEVICE)
else:
    model = Seq2SeqWithEmbeddingmodClass(num_encoder_layers=num_encoder_layers,
                                num_decoder_layers=num_decoder_layers,
                                input_size=train_dataset.shape[-1],
                                emb_size=emb_size,
                                nhead=nhead,
                                dim_feedforward=dim,
                                dropout=0,
                                num_classes=num_classes).to(DEVICE)

print(sum(p.numel() for p in model.parameters() if p.requires_grad))

torch.cuda.empty_cache()
gc.collect()
opt = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(adam_beta1, adam_beta2))
train_dataset = train_dataset.to(DEVICE)
print(train_dataset.shape)

# # trained_model, loss_traj = train_model(model, train_dataset, opt, prediction_len=PREDICTION_LENGTH, num_epochs=2000, device=DEVICE, checkpoint_suffix=save_name)
if not weighted:
    train_model_vocab_autoreg(model, train_dataset, opt, prediction_len=PREDICTION_LENGTH, num_epochs=NUM_EPOCHS, device=DEVICE, checkpoint_suffix=save_name, num_classes=num_classes, vocab_dict=vocab_dict)
else:
    weights = np.ones(PREDICTION_LENGTH)
    # weights[0:9] = np.arange(1,10,1)[::-1]
    # weights[-9:] = np.arange(1,10,1)
    # print(weights)
    weights = 1/sum(weights)*weights
    train_model_vocab(model, train_dataset, opt, weights, prediction_len=PREDICTION_LENGTH, num_epochs=NUM_EPOCHS, device=DEVICE, checkpoint_suffix=save_name, lw=loss_weight, alpha=alpha)
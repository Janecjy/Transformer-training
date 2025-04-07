import torch
from models import Seq2SeqWithEmbeddingmodClass, Seq2SeqWithEmbeddingmodClassMoreEmbedding
from utils import train_model_vocab_single, train_model_vocab_multi
import pickle
import argparse
import numpy as np
import os
import ntpath

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
# parser.add_argument('--Dataset', '-Data', help='Filename prefix for Training Dataset in the NewDatasets Folder', type=str, default='FullDataset')
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
parser.add_argument('--ModelName', '-M', help='Name of the model', type=str, default='BaseTransformer3')
parser.add_argument('--boundaries-file', '-BF', help='Path to the boundary file used for tokenization', default="/datastor1/janec/datasets/boundaries/boundaries-quantile100.pkl")
parser.add_argument('--token-type', '-TT', choices=['single','multi'], default='single',
                    help='Choose single combo token or multi-head token approach')

args = parser.parse_args()
# dataset_name = args.Dataset 
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
add_name = args.ModelName
save_name = add_name+"_Transformer3_"+str(dim)+"_"+str(num_encoder_layers)+"_"+str(num_decoder_layers)+"_"+str(emb_size)+"_"+str(nhead)+"_lr_"+str(learning_rate)+"_"+str(args.boundaries_file)+"_"+str(args.token_type)
more_embedding = args.MoreEmbedding

#CONSTANTS
DEVICE = torch.device("cuda:"+str(gpu) if torch.cuda.is_available() else "cpu")
import gc
torch.cuda.empty_cache()
gc.collect()
# DEVICE = torch.device("cpu")
print(DEVICE)
PAD_IDX = 2
BATCH_SIZE = 1024*2
NUM_EPOCHS = 1000
CONTEXT_LENGTH = 10
PREDICTION_LENGTH = 10

boundary_base = ntpath.basename(args.boundaries_file)  # e.g. boundaries-quantile100.pkl
boundary_noext = os.path.splitext(boundary_base)[0]    # e.g. boundaries-quantile100

if args.token_type == 'single':
    tokenized_path = os.path.join(
        "/datastor1/janec/datasets/combined",
        f"{boundary_noext}-tokenized-single.pkl"
    )
else:
    tokenized_path = os.path.join(
        "/datastor1/janec/datasets/combined",
        f"{boundary_noext}-tokenized-multi.pkl"
    )

if not os.path.isfile(tokenized_path):
    raise FileNotFoundError(f"Tokenized file not found: {tokenized_path}")

with open(tokenized_path, "rb") as f:
    tokenized_data = pickle.load(f)


# tokenized_data might contain different keys, depending on single vs multi
# single => { base_rtt, tokens_single, combo_dict, combo_list }
# multi  => { base_rtt, tokens_multi, boundaries_dict }

base_rtt = tokenized_data["base_rtt"]  # shape (N, 20)

if args.token_type == 'single':
    train_dataset_np = tokenized_data["tokens_single"]  # shape (N, 20)
    combo_dict = tokenized_data["combo_dict"]           # for the classification
    num_classes = len(combo_dict)
    print(f"[Single Combo] Loaded shape = {train_dataset_np.shape}, vocab size={num_classes}")
    train_dataset_np = np.expand_dims(train_dataset_np, axis=-1)
else:
    train_dataset_np = tokenized_data["tokens_multi"]   # shape (N,20,5)
    num_classes = int(np.max(train_dataset_np) + 1)
    print(f"[Multi-Head] shape={train_dataset_np.shape}, max bucket index={num_classes-1}")


train_dataset = torch.from_numpy(train_dataset_np).to(torch.float32)

print("Final train_dataset shape:", train_dataset.shape)

train_dataset = train_dataset.to(DEVICE)
# Possibly check dataset size
print("Train data loaded: shape =", train_dataset.shape)

# We define the "vocab_dict" only if single
if args.token_type == 'single':
    vocab_dict = combo_dict
else:
    vocab_dict = None  # or some dictionary if you want a multi approach
    
model = Seq2SeqWithEmbeddingmodClass(num_encoder_layers=num_encoder_layers,
                            num_decoder_layers=num_decoder_layers,
                            input_size=train_dataset.shape[-1],
                            emb_size=emb_size,
                            nhead=nhead,
                            dim_feedforward=dim,
                            dropout=0,
                            num_classes=num_classes).to(DEVICE)

print(sum(p.numel() for p in model.parameters() if p.requires_grad))

opt = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(adam_beta1, adam_beta2))
if args.token_type == "single":
    # Single-combo
    model, loss_traj = train_model_vocab_single(
        model,
        dataset=train_dataset,
        optimizer=opt,
        prediction_len=PREDICTION_LENGTH,
        device=DEVICE,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        checkpoint_suffix=save_name,
        num_classes=num_classes,
    )
else:
    # Multi-head
    # We assume 5 heads. Possibly define #heads by last dimension? 
    # Or pass a param. We'll do 5 to match shape (N,20,5).
    model, loss_traj = train_model_vocab_multi(
        model,
        dataset=train_dataset,
        optimizer=opt,
        prediction_len=PREDICTION_LENGTH,
        device=DEVICE,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
        checkpoint_suffix=save_name,
        num_heads=5,
        max_bucket=num_classes,
    )

print("Training complete!")
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from utils import *
from models.vqvae import VQVAE
from torch.autograd import Variable
from torchvision.utils import save_image
import os
parser = argparse.ArgumentParser()

"""
Hyperparameters
"""
parser.add_argument("--n_hiddens", type=int, default=128)
parser.add_argument("--n_residual_hiddens", type=int, default=32)
parser.add_argument("--n_residual_layers", type=int, default=2)
parser.add_argument("--embedding_dim", type=int, default=64)
parser.add_argument("--n_embeddings", type=int, default=512)
parser.add_argument("--beta", type=float, default=.25)
parser.add_argument("--loadpth",  type=str, default='./results/vqvae_data_tst2.pth')
parser.add_argument("--data_dir",  type=str, default='/home/karam/Downloads/bco/')
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = VQVAE(args.n_hiddens, args.n_residual_hiddens,
              args.n_residual_layers, args.n_embeddings, args.embedding_dim, args.beta).to(device)
assert args.loadpth is not ''
model.load_state_dict(torch.load(args.loadpth)['model'])
model.eval()
print("Loaded model")

save_dir=os.getcwd() + '/data'
data_dir = args.data_dir
data1 = np.load(data_dir+"/bcov5_0.npy")
data2 = np.load(data_dir+"/bcov5_1.npy")
data3 = np.load(data_dir+"/bcov5_2.npy")
data = np.concatenate((data1,data2,data3),axis=0)
n_trajs = len(data)
trjs, length = data.shape[:2]
print("Loaded data")

latents,ctxts=[],[]
with torch.no_grad():
    for tr in range(trjs):
        x,c = get_torch_images_from_numpy(data[tr, :], True ,normalize=True)
        x,c = x.to(device), c.to(device)
        encoding_indices_x = model(x,latent_only=True)
        encoding_indices_c = model(c,latent_only=True)
        latents.append(encoding_indices_x.detach().cpu().numpy().squeeze().reshape(length,-1))
        ctxts.append(encoding_indices_c.detach().cpu().numpy().squeeze().reshape(length,-1))

np.save(save_dir+'/bco1_xlatents.npy',latents)
np.save(save_dir+'/bco1_clatents.npy',ctxts)
print(np.array(latents).shape)
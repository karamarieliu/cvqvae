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
parser.add_argument("--loadpth",  type=str, default='./results/vqvae_data_bo.pth')
parser.add_argument("--data_dir",  type=str, default='/home/karam/Downloads/bco/')
parser.add_argument("--data",  type=str, default='bco')
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Load model
model = VQVAE(args.n_hiddens, args.n_residual_hiddens,
              args.n_residual_layers, args.n_embeddings, args.embedding_dim, args.beta).to(device)
assert args.loadpth is not ''
model.load_state_dict(torch.load(args.loadpth)['model'])
model.eval()
print("Loaded model")


#Load data
save_dir=os.getcwd() + '/data'
data_dir = args.data_dir
if args.data=='bco':
    data1 = np.load(data_dir+"/bcov5_0.npy")
    data2 = np.load(data_dir+"/bcov5_1.npy")
    data3 = np.load(data_dir+"/bcov5_2.npy")
    data4 = np.load(data_dir+"/bcov5_3.npy")
    data = np.concatenate((data1,data2,data3,data4),axis=0)
elif args.data=='bo':
    data = np.load(data_dir+"/bo-150-50-20.npy",allow_pickle=True)

n_trajs, length = data.shape[:2]
print("Loaded data")

latents,ctxts=[],[]
with torch.no_grad():
    for tr in range(n_trajs):
        x,c = get_torch_images_from_numpy(data[tr, :], True ,normalize=True)
        x,c = x.to(device),c.to(device)
        encoding_indices_x = model(x,latent_only=True)
        encoding_indices_c = model(c,latent_only=True)
        latents.append(encoding_indices_x.detach().cpu().numpy().squeeze().reshape(length,-1))
        ctxts.append(encoding_indices_c.detach().cpu().numpy().squeeze().reshape(length,-1))

np.save(save_dir+'/%s_xlatents.npy'%args.data,latents)
np.save(save_dir+'/%s_clatents.npy'%args.data,ctxts)
print("Generated image indices with shape ", np.array(latents).shape)


# with torch.no_grad():
#     for tr in range(0,n_trajs,30):
#         _,c = get_torch_images_from_numpy(data[tr, :], True ,normalize=True)
#         c=c[0][None,:].to(device)
#         vec_c = model(torch.cat((c,c),dim=1),vec_only=True)
#         ctxts.append(vec_c.detach().cpu().numpy().squeeze().reshape(1,64,16,16))

# np.save(save_dir+'/bco_zgivenc_cvec.npy',ctxts)
# print("Generated context vectors with shape ", np.array(ctxts).shape)
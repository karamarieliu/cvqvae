import torch
import torch.nn as nn
from torchvision import datasets, transforms
import numpy as np
from torchvision.utils import save_image
import time
import os 
import sys
from torch.autograd import Variable
from torchvision.utils import save_image
import os
# from tensorboard_logger import log_value
# from tensorboard_logger import configure
from utils import *
current_dir = sys.path.append(os.getcwd())
from models.pixelcnn import GatedPixelCNN, GatedPixelCNN_Snail
import utils
from models.vqvae import VQVAE

"""
Hyperparameters
"""
import argparse 
parser = argparse.ArgumentParser()
parser.add_argument("--n_residual_layers", type=int, default=2)
parser.add_argument("--embedding_dim", type=int, default=64)
parser.add_argument("--beta", type=float, default=.25)
parser.add_argument("--n_residual_hiddens", type=int, default=32)
parser.add_argument("--n_hiddens", type=int, default=128)
parser.add_argument("--conditional", default=True)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--log_interval", type=int, default=500)
parser.add_argument("--sample_interval", type=int, default=2000)
parser.add_argument("--img_dim", type=int, default=16)
parser.add_argument("--input_dim", type=int, default=1,
    help='1 for grayscale 3 for rgb')
parser.add_argument("--n_embeddings", type=int, default=512,
    help='number of embeddings from VQ VAE')
parser.add_argument("--n_layers", type=int, default=8)
parser.add_argument("--learning_rate", type=float, default=1e-3)
parser.add_argument("--loadpth_vq",  type=str, default='./results/vqvae_data_4files_1.pth')
parser.add_argument("--loadpth_pcnn",  type=str, default='')
parser.add_argument("--prefix",  type=str, default='')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# configure('./results/var_log', flush_secs=5)
img_dim=args.img_dim
data = np.load("./data/bco_xlatents.npy")
data,val=data[30*10:],data[:30*10]
context = np.load("./data/bco_clatents.npy").squeeze()
context,valcon=context[30*10:],context[:30*10]
sample_c_imgs = np.load("./data/bco_tstcon_40.npy")
sample_c_imgs = get_torch_images_from_numpy(sample_c_imgs, True, one_image=True)
data,val=data.reshape(-1,256),val.reshape(-1,256)
context,valcon=context.reshape(-1,256),context.reshape(-1,256)
n_trajs, length = data.shape[:2]

model = GatedPixelCNN_Snail(n_embeddings=args.n_embeddings, imgximg=args.img_dim**2, 
    n_layers=args.n_layers, conditional=args.conditional).to(device)
model.train()
criterion = nn.CrossEntropyLoss().cuda()
opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

if args.loadpth_vq is not '':
    vae = VQVAE(args.n_hiddens, args.n_residual_hiddens,
              args.n_residual_layers, args.n_embeddings, args.embedding_dim, args.beta).cuda()
    vae.load_state_dict(torch.load(args.loadpth_vq)['model'])
    print("VQ Loaded")
    vae.eval()
    sample_c=vae(sample_c_imgs,latent_only=True).detach().cpu().numpy().reshape(-1,length).squeeze()

# 
if args.loadpth_pcnn is not '':
    model.load_state_dict(torch.load(args.loadpth_pcnn))
    print("PCNN Loaded")

n_trajs = len(data)
dt = n_trajs // context.shape[0]
n_batch = int(n_trajs / args.batch_size)
n_trajs_t = len(val)
dv = n_trajs_t // valcon.shape[0]
n_batch_t = int(n_trajs_t / args.batch_size)

# 
"""
train, test, and log
"""

def train(epoch):
    train_loss = []
    log_interval=min(args.log_interval,n_batch-2)
    sample_interval=min(args.sample_interval,n_batch-2)
    print(n_batch)
    start=time.time()    
    for it in range(n_batch):

        idx = np.random.choice(n_trajs, size=args.batch_size)
        x = from_numpy_to_var(data[idx].reshape(args.batch_size,img_dim,img_dim),dtype='long').long().cuda()
        c = from_numpy_to_var(context[idx].reshape(args.batch_size,img_dim,img_dim),dtype='long').long().cuda()
        
        #if c = vq vectors
        # c = from_numpy_to_var(context[idx//dt].reshape(args.batch_size,args.embedding_dim,img_dim,img_dim)).cuda()
        # _,hat,_=vae(min_encoding_indices=x)                                                                                         
        # save_image(hat,'/home/karam/Downloads/xhat.png')     
        # _,chat,_=vae(z_q=c)
        # save_image(chat,'/home/karam/Downloads/chat.png')     

        # Train PixelCNN with images
        logits = model(x,c=c)
        logits = logits.permute(0, 2, 3, 1).contiguous()
        loss = criterion(logits.view(-1, args.n_embeddings),x.view(-1))

        opt.zero_grad()
        loss.backward()
        opt.step()

        train_loss.append(loss.item())

        if it % log_interval == 0 and it > 0:
            print('\tIter: [{}/{} ({:.2f}%)]\tLoss: {:.6f}\n\t\tMinutes: {:.2f} \tHr/ep:{:.2f}'.format(
                it,n_batch ,
                100*it / n_batch,
                np.asarray(train_loss)[-log_interval:].mean(0),
                (time.time()-start)/60,
                n_batch/it*(time.time()-start)/(60*60)
            ))
            # log_value('tr_loss', loss.item(), it + n_batch*epoch)
        if it % sample_interval == 0 and it > 0: 
            torch.save(model.state_dict(), 'results/bco_pixelcnn_%s_%d.pth'%(args.prefix,epoch))
            generate_samples(epoch)


def test(epoch):
    val_loss = []
    with torch.no_grad():
        for it in range(n_batch_t):

            idx = np.random.choice(n_trajs_t, size=args.batch_size)
            x = from_numpy_to_var(val[idx].reshape(args.batch_size,img_dim,img_dim)).long().cuda()
            c = from_numpy_to_var(valcon[idx].reshape(args.batch_size,img_dim,img_dim)).long().cuda()

            logits = model(x, c=c)
            logits = logits.permute(0, 2, 3, 1).contiguous()
            loss = criterion( logits.view(-1, args.n_embeddings),x.view(-1))
            
            val_loss.append(loss.item())

    avval=np.asarray(val_loss).mean(0)
    print('Validation Completed!\tLoss: {}'.format(avval))
    # log_value('val_loss', avval.item(), n_batch_t*epoch)

    return np.asarray(val_loss).mean(0)


def generate_samples(epoch):
    n_samples, ctxts_sampled=10,1
    labels = torch.zeros(n_samples).long().cuda()

    #Test samples
    idx = np.random.choice(sample_c.shape[0], size=ctxts_sampled)
    con = from_numpy_to_var(sample_c[idx]).repeat(n_samples,1,1,1).reshape(n_samples,img_dim,img_dim).long().cuda()
    x_tilde = model.generate(labels, c=con, shape=(img_dim,img_dim), batch_size=n_samples)

    if args.loadpth_vq is not '':
        _, x_hat, _ = vae(min_encoding_indices=x_tilde)
        save_image(torch.cat((sample_c_imgs[idx],x_hat),dim=0),'./results/%s_testsamples_%d.png'%(args.prefix,epoch),nrow=n_samples+1)
    
    #Training samples 
    idx = np.random.choice(n_trajs, size=n_samples)
    c = from_numpy_to_var(context[idx].reshape(n_samples,img_dim,img_dim)).long().cuda()
    x_tilde = model.generate(labels, c=c, shape=(img_dim,img_dim), batch_size=n_samples)
    
    if args.loadpth_vq is not '':
        _,x_hat_con,_=vae(min_encoding_indices=from_numpy_to_var(context[idx]).long())
        _, x_hat, _ = vae(min_encoding_indices=x_tilde)
        save_image(torch.cat((x_hat_con,x_hat),dim=0),'./results/%s_trsamples_%d.png'%(args.prefix,epoch),nrow=n_samples)

BEST_LOSS = 999
LAST_SAVED = -1
for epoch in range(0, args.epochs):

    print("\n######### Epoch {}:".format(epoch))
    train(epoch)
    cur_loss = test(epoch)

    if cur_loss <= BEST_LOSS:
        BEST_LOSS = cur_loss
        LAST_SAVED = epoch

        # print("Saving model!")
        # torch.save(model.state_dict(), 'results/bco_pixelcnn_%s.pth'%args.prefix)

    print("Lowest loss: {}".format(LAST_SAVED))
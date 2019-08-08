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
from tensorboard_logger import log_value
from tensorboard_logger import configure

def from_numpy_to_var(npx, dtype='float32'):
    var = Variable(torch.from_numpy(npx.astype(dtype)))
    if torch.cuda.is_available():
        return var.cuda()
    else:
        return var
"""
add vqvae and pixelcnn dirs to path
make sure you run from vqvae directory
"""
current_dir = sys.path.append(os.getcwd())

from models.pixelcnn import GatedPixelCNN
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

parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--log_interval", type=int, default=1000)
parser.add_argument("--img_dim", type=int, default=16)
parser.add_argument("--input_dim", type=int, default=1,
    help='1 for grayscale 3 for rgb')
parser.add_argument("--n_embeddings", type=int, default=512,
    help='number of embeddings from VQ VAE')
parser.add_argument("--n_layers", type=int, default=15)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--loadpth_vq",  type=str, default='./results/vqvae_data_tst2.pth')
parser.add_argument("--loadpth_pcnn",  type=str, default='./results/bco_pixelcnn.pth')
args = parser.parse_args()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
configure('./results/var_log', flush_secs=5)
img_dim=args.img_dim
data = np.load("./data/"+"/bco_latents.npy")
data,val = data[30000:],data[:30000]
model = GatedPixelCNN(args.n_embeddings, args.img_dim**2, args.n_layers).to(device)
model.train()
criterion = nn.CrossEntropyLoss().cuda()
opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
data1 = np.load(data_dir+"/bcov5_0.npy")
data2 = np.load(data_dir+"/bcov5_1.npy")
data3 = np.load(data_dir+"/bcov5_2.npy")
contexts = np.concatenate((data1,data2,data3),axis=0)

if args.loadpth_vq is not '':
    vae = VQVAE(args.n_hiddens, args.n_residual_hiddens,
              args.n_residual_layers, args.n_embeddings, args.embedding_dim, args.beta).cuda()
    vae.load_state_dict(torch.load(args.loadpth_vq)['model'])
    print("Loaded")
    vae.eval()
if args.loadpth_pcnn is not '':
    model.load_state_dict(torch.load(args.loadpth_pcnn))



n_trajs = len(data)
n_batch = int(n_trajs / args.batch_size)

n_trajs_t = len(val)
n_batch_t = int(n_trajs_t / args.batch_size)

"""
train, test, and log
"""

def train(epoch):
    train_loss = []

    for it in range(n_batch):
        start_time = time.time()
        idx = np.random.choice(n_trajs, size=args.batch_size)
        x = from_numpy_to_var(data[idx].reshape(args.batch_size,img_dim,img_dim),dtype='long').long().cuda()
        label=torch.zeros(args.batch_size).long().cuda()

        # Train PixelCNN with images
        logits = model(x, label)
        logits = logits.permute(0, 2, 3, 1).contiguous()

        loss = criterion(
            logits.view(-1, args.n_embeddings),
            x.view(-1)
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        train_loss.append(loss.item())

        if it % args.log_interval == 0 and it > 0:
            print('\tIter: [{}/{} ({:.0f}%)]\tLoss: {} Time: {}'.format(
                it,n_batch ,
                it*batch_size / n_batch,
                np.asarray(train_loss)[-args.log_interval:].mean(0),
                time.time() - start_time
            ))
            log_value('tr_loss', loss.item(), it + n_batch*epoch)



def test(epoch):
    start_time = time.time()
    val_loss = []
    with torch.no_grad():
        for it in range(n_batch_t):

            idx = np.random.choice(n_trajs_t, size=args.batch_size)
            x = from_numpy_to_var(val[idx].reshape(args.batch_size,img_dim,img_dim),dtype='long').long().cuda()
            label=torch.zeros(args.batch_size).long().cuda()

            logits = model(x, label)
            logits = logits.permute(0, 2, 3, 1).contiguous()
            loss = criterion(
                logits.view(-1, args.n_embeddings),
                x.view(-1)
            )
            
            val_loss.append(loss.item())

    avval=np.asarray(val_loss).mean(0)
    print('Validation Completed!\tLoss: {} Time: {}'.format(
        avval,
        time.time() - start_time
    ))
    log_value('val_loss', avval.item(), n_batch_t*epoch)

    return np.asarray(val_loss).mean(0)


def generate_samples(epoch):
    n_samples=32
    # label = torch.arange(10).expand(10, 10).contiguous().view(-1)
    label=torch.zeros(n_samples)
    label = label.long().cuda()[:n_samples]

    x_tilde = model.generate(label, shape=(img_dim,img_dim), batch_size=n_samples)
    if args.loadpth_vq is not '':
        _, x_hat, _ = vae(min_encoding_indices=x_tilde)
        save_image(x_hat,'./results/samples_%d.png'%epoch)



BEST_LOSS = 999
LAST_SAVED = -1
for epoch in range(0, args.epochs):
    generate_samples(epoch)

    print("\n######### Epoch {}:".format(epoch))
    train(epoch)
    cur_loss = test(epoch)

    if cur_loss <= BEST_LOSS:
        BEST_LOSS = cur_loss
        LAST_SAVED = epoch

        print("Saving model!")
        torch.save(model.state_dict(), 'results/bco_pixelcnn.pth')
    else:
        print("Not saving model! Last saved: {}".format(LAST_SAVED))
    generate_samples(epoch)

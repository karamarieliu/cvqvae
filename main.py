import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import utils
from models.vqvae import VQVAE
from torch.autograd import Variable
from torchvision.utils import save_image
import os
parser = argparse.ArgumentParser()


"""
Hyperparameters
"""
timestamp = utils.readable_timestamp()

parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_updates", type=int, default=5000)
parser.add_argument("--n_hiddens", type=int, default=128)
parser.add_argument("--n_residual_hiddens", type=int, default=32)
parser.add_argument("--n_residual_layers", type=int, default=2)
parser.add_argument("--embedding_dim", type=int, default=64)
parser.add_argument("--n_embeddings", type=int, default=512)
parser.add_argument("--beta", type=float, default=.25)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--log_interval", type=int, default=1000)
parser.add_argument("--dataset",  type=str, default='CIFAR10')
parser.add_argument("--loadpth",  type=str, default='./results/vqvae_data_tst2.pth')

# whether or not to save model
parser.add_argument("--filename",  type=str, default=timestamp)

args = parser.parse_args()
args.save = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.save:
    print('Results will be saved in ./results/vqvae_' + args.filename + '.pth')

"""
Load data and define batch data loaders
"""

# training_data, validation_data, training_loader, validation_loader, x_train_var = utils.load_data_and_data_loaders(
#     args.dataset, args.batch_size)
"""
Set up VQ-VAE model with components defined in ./models/ folder
"""

model = VQVAE(args.n_hiddens, args.n_residual_hiddens,
              args.n_residual_layers, args.n_embeddings, args.embedding_dim, args.beta).to(device)
if args.loadpth is not '':
    model.load_state_dict(torch.load(args.loadpth)['model'])
    print("Loaded")

"""
Set up optimizer and training loop
"""
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)
data_dir = '/home/karam/Downloads/bco/'
model.train()

results = {
    'n_updates': 0,
    'recon_errors': [],
    'loss_vals': [],
    'perplexities': [],
}
data1 = np.load(data_dir+"/bcov5_0.npy")
data2 = np.load(data_dir+"/bcov5_1.npy")
data3 = np.load(data_dir+"/bcov5_2.npy")
data = np.concatenate((data1,data2,data3),axis=0)
# d=data[:,:,0]
# newnpy=[]
# n_trajs,traj_len = d.shape
# for i in range(n_trajs):
#     img=d[i,0][:,:,:3]/255
#     newnpy.append(img)

x_train_var = 0.02694245912284954
n_trajs = len(data)
data_size = sum([len(data[i]) - 1 for i in range(n_trajs)])
n_batch = int(data_size / args.batch_size)
save_dir=os.getcwd() + '/results'

def train():

    for i in range(args.n_updates):
        print(n_batch)
        for it in range(n_batch):
            idx = np.random.choice(n_trajs, size=args.batch_size)
            t = np.array([np.random.randint(len(data[i]) - 1) for i in idx])            
            x,_ = get_torch_images_from_numpy(data[idx, t], True ,normalize=True)
            x = x.to(device)
            optimizer.zero_grad()

            embedding_loss, x_hat, perplexity = model(x)
            recon_loss = torch.mean((x_hat - x)**2) / x_train_var
            loss = recon_loss + embedding_loss

            loss.backward()
            optimizer.step()

            results["recon_errors"].append(recon_loss.cpu().detach().numpy())
            results["perplexities"].append(perplexity.cpu().detach().numpy())
            results["loss_vals"].append(loss.cpu().detach().numpy())
            results["n_updates"] = i

            if it % args.log_interval == 0:
                """
                save model and print values
                """
                if args.save:
                    hyperparameters = args.__dict__
                    utils.save_model_and_results(
                        model, results, hyperparameters, args.filename)

                print('Update #', i, 'Recon Error:',
                      np.mean(results["recon_errors"][-args.log_interval:]),
                      'Loss', np.mean(results["loss_vals"][-args.log_interval:]),
                      'Perplexity:', np.mean(results["perplexities"][-args.log_interval:]))

        save_image(torch.cat((x[:10],x_hat[:10]),dim=0),save_dir+"recon_%d_%d.png"%(i,it),nrow=10)

if __name__ == "__main__":
    train()

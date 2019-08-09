
import torch
import torch.nn as nn
import numpy as np
from models.encoder import Encoder
from models.quantizer import VectorQuantizer
from models.decoder import Decoder


class VQVAE(nn.Module):
    def __init__(self, h_dim, res_h_dim, n_res_layers,
                 n_embeddings, embedding_dim, beta, save_img_embedding_map=False):
        super(VQVAE, self).__init__()
        # encode image into continuous latent space
        self.encoder = Encoder(3, h_dim, n_res_layers, res_h_dim)
        self.pre_quantization_conv = nn.Conv2d(h_dim, embedding_dim, kernel_size=1, stride=1)
        # pass continuous latent vector through discretization bottleneck
        self.vector_quantization = VectorQuantizer(n_embeddings, embedding_dim, beta)
        # decode the discrete latent representation
        self.decoder = Decoder(embedding_dim, h_dim, n_res_layers, res_h_dim)

        if save_img_embedding_map:
            self.img_to_embedding_map = {i: [] for i in range(n_embeddings)}
        else:
            self.img_to_embedding_map = None

    def forward(self, x=None, verbose=False,latent_only=False, vec_only=False, min_encoding_indices=None):
        # latent_only: flagged if training PixelCNN and need 
        # to save the latent space representations 
        
        # min_encoding_indices: provided if want to decode samples sampled from 
        # trained PixelCNN
        
        if min_encoding_indices is None:
            assert x is not None
            z_e = self.encoder(x)

            # if vec_only:
                # return z_e 
            z_e = self.pre_quantization_conv(z_e)

            embedding_loss, z_q, perplexity, _, indices = self.vector_quantization(z_e)

            if latent_only:
                return indices
        else:
            z_q = self.vector_quantization(min_encoding_indices=min_encoding_indices)
            embedding_loss, perplexity = 0,0

        if vec_only:
            return z_q
            
        x_hat = self.decoder(z_q)

        if verbose and min_encoding_indices is None:
            print('original data shape:', x.shape)
            print('encoded data shape:', z_e.shape)
            print('recon data shape:', x_hat.shape)
            assert False

        return embedding_loss, x_hat, perplexity


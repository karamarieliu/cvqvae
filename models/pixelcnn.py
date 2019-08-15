
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
from functools import partial, lru_cache


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            pass

def shift_down(input, size=1):
    return F.pad(input, [0, 0, size, 0])[:, :, : input.shape[2], :]

def shift_right(input, size=1):
    return F.pad(input, [size, 0, 0, 0])[:, :, :, : input.shape[3]]

class WNConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        padding=0,
        bias=True,
        activation=None,
    ):
        super().__init__()

        self.conv = nn.utils.weight_norm(
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=bias,
            )
        )

        self.out_channel = out_channel

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]

        self.kernel_size = kernel_size

        self.activation = activation

    def forward(self, input):
        out = self.conv(input)

        if self.activation is not None:
            out = self.activation(out)

        return out

class GatedResBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        channel,
        kernel_size,
        conv='wnconv2d',
        activation=nn.ELU,
        dropout=0.1,
        auxiliary_channel=0,
        condition_dim=0,
    ):
        super().__init__()

        if conv == 'wnconv2d':
            conv_module = partial(WNConv2d, padding=kernel_size // 2)

        elif conv == 'causal_downright':
            conv_module = partial(CausalConv2d, padding='downright')

        elif conv == 'causal':
            conv_module = partial(CausalConv2d, padding='causal')

        self.activation = activation(inplace=True)
        self.conv1 = conv_module(in_channel, channel, kernel_size)

        if auxiliary_channel > 0:
            self.aux_conv = WNConv2d(auxiliary_channel, channel, 1)

        self.dropout = nn.Dropout(dropout)

        self.conv2 = conv_module(channel, in_channel * 2, kernel_size)

        if condition_dim > 0:
            # self.condition = nn.Linear(condition_dim, in_channel * 2, bias=False)
            self.condition = WNConv2d(condition_dim, in_channel * 2, 1, bias=False)

        self.gate = nn.GLU(1)

    def forward(self, input, aux_input=None, condition=None):
        out = self.conv1(self.activation(input))

        if aux_input is not None:
            out = out + self.aux_conv(self.activation(aux_input))

        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)

        if condition is not None:
            condition = self.condition(condition)
            out += condition
            # out = out + condition.view(condition.shape[0], 1, 1, condition.shape[1])

        out = self.gate(out)
        out += input

        return out

class CondResNet(nn.Module):
    def __init__(self, in_channel, channel, kernel_size, n_res_block):
        super().__init__()

        blocks = [WNConv2d(in_channel, channel, kernel_size, padding=kernel_size // 2)]

        for i in range(n_res_block):
            blocks.append(GatedResBlock(channel, channel, kernel_size))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)

class GatedActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, con_f=None, con_g=None):
        x, y = x.chunk(2, dim=1)
        if con_f is None:
            return F.tanh(x) * F.sigmoid(y)
        return F.tanh(x+con_f) * F.sigmoid(y+con_g)

class CausalConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride=1,
        padding='downright',
        activation=None,
    ):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * 2

        self.kernel_size = kernel_size

        if padding == 'downright':
            pad = [kernel_size[1] - 1, 0, kernel_size[0] - 1, 0]

        elif padding == 'down' or padding == 'causal':
            pad = kernel_size[1] // 2

            pad = [pad, pad, kernel_size[0] - 1, 0]

        self.causal = 0
        if padding == 'causal':
            self.causal = kernel_size[1] // 2

        self.pad = nn.ZeroPad2d(pad)

        self.conv = WNConv2d(
            in_channel,
            out_channel,
            kernel_size,
            stride=stride,
            padding=0,
            activation=activation,
        )

    def forward(self, input):
        out = self.pad(input)

        if self.causal > 0:
            self.conv.conv.weight_v.data[:, :, -1, self.causal :].zero_()

        out = self.conv(out)

        return out

class GatedMaskedConv2d(nn.Module):
    def __init__(self, mask_type, imgximg, kernel, residual=True, n_classes=10, 
        x_one_hot=0,c_one_hot=2):
        super().__init__()
        assert kernel % 2 == 1, print("Kernel size must be odd")
        self.mask_type = mask_type
        self.residual = residual
        self.c_one_hot=c_one_hot
        self.x_one_hot=x_one_hot

        if c_one_hot==2:
            cdim=3*imgximg #one hot encoding + embedding for indices
        elif c_one_hot==1: 
            cdim=2*imgximg  #one hot encoding
        else:
            cdim=imgximg #embedding for indices  
        if x_one_hot==1:
            dim=2*imgximg
        else:
            dim=imgximg
        if c_one_hot==3:
            dim*=2
        kernel_shp = (kernel // 2 + 1, kernel)  # (ceil(n/2), n)
        padding_shp = (kernel // 2, kernel // 2)
        self.vert_stack = nn.Conv2d(
            dim, dim * 2,
            kernel_shp, 1, padding_shp
        )
        self.vert_to_horiz = nn.Conv2d(2 * dim, 2 * dim, 1)
        self.vert_to_horiz_c = nn.Conv2d(dim, dim, 1)
        kernel_shp = (1, kernel // 2 + 1)
        padding_shp = (0, kernel // 2)
        self.horiz_stack = nn.Conv2d(
            dim, dim * 2,
            kernel_shp, 1, padding_shp
        )
        self.horiz_resid = nn.Conv2d(dim, dim, 1)
        self.con_f = nn.Conv2d(cdim, dim, 1)
        self.con_g = nn.Conv2d(cdim, dim, 1)
        self.gate = GatedActivation()

    def make_causal(self):
        self.vert_stack.weight.data[:, :, -1].zero_()  # Mask final row
        self.horiz_stack.weight.data[:, :, :, -1].zero_()  # Mask final column

    def forward(self, x_v, x_h, c):
        if c is not None:
            con_f = self.con_f(c)
            con_g = self.con_g(c)
        else:
            con_f,con_g = None,None

        if self.mask_type == 'A':
            self.make_causal()

        h_vert = self.vert_stack(x_v)
        h_vert = h_vert[:, :, :x_v.size(-1), :]
        out_v = self.gate(h_vert, con_f, con_g)

        h_horiz = self.horiz_stack(x_h)
        h_horiz = h_horiz[:, :, :, :x_h.size(-2)]
        v2h = self.vert_to_horiz(h_vert)

        out = self.gate(v2h + h_horiz, con_f, con_g)
        if self.residual:
            out_h = self.horiz_resid(out) + x_h
        else:
            out_h = self.horiz_resid(out)

        return out_v, out_h


class GatedPixelCNN(nn.Module):
    def __init__(self, n_embeddings, imgximg, n_layers, conditional=True, 
            n_cond_res_block=3, c_one_hot=2, x_one_hot=0):
        super().__init__()
        if c_one_hot==2:
            self.dim=3*imgximg #one hot encoding + embedding for indices
        elif c_one_hot==1: 
            self.dim=2*imgximg  #one hot encoding
        else:
            self.dim=imgximg #embedding for indices   
        if x_one_hot == 1:
            outdim=2*imgximg
        else:
            outdim=imgximg
        if c_one_hot==3:
            outdim*=2     
        self.conditional=conditional
        self.c_one_hot=c_one_hot
        self.x_one_hot=x_one_hot
        self.embedding = nn.Embedding(n_embeddings, imgximg)        
        self.n_class=n_embeddings
        self.layers = nn.ModuleList()


        for i in range(n_layers):
            mask_type = 'A' if i == 0 else 'B'
            kernel = 7 if i == 0 else 3
            residual = False if i == 0 else True

            self.layers.append(
                GatedMaskedConv2d(mask_type, imgximg, kernel, residual,
                    c_one_hot=c_one_hot,x_one_hot=x_one_hot)
            )

        # Add the output layer
        out = []
        cond_res_kernel=5
        self.out = nn.Sequential(
            nn.Conv2d(outdim, 2*outdim, 1),         
            nn.ReLU(True),
            nn.Conv2d(2*outdim, n_embeddings, 1)  
            )          

        if n_cond_res_block > 0:
            self.cond_resnet = CondResNet(
                self.dim, self.dim, cond_res_kernel, n_cond_res_block
            )

        self.apply(weights_init)



    def forward(self, x, c):  
        # one hot: 0 = don't use at all
        # 1: use just one hot
        # 2: cat c w one hot
        shp = x.size() + (-1, )
        c_one_hot=self.c_one_hot
        x_one_hot=self.x_one_hot
        if x_one_hot==1:
            x = F.one_hot(x, self.n_class).permute(0, 3, 1, 2).float()
        elif x_one_hot==2:
            print(1/0)
        else:
            x = self.embedding(x.view(-1)).view(shp)  # (B, H, W, C)
            x = x.permute(0, 3, 1, 2) # (B, C, W, H)     

        if c is not None:
            if self.c_one_hot==3:
                c_one_hot=x_one_hot
            if c_one_hot==1:
                c = F.one_hot(c, self.n_class)
                c = c.permute(0, 3, 1, 2)
            elif c_one_hot==2:
                c1 = F.one_hot(c, self.n_class)
                c1 = c1.permute(0, 3, 1, 2).float()
                c2 = self.embedding(c.view(-1)).view(shp)  # (B, H, W, C)
                c2 = c2.permute(0, 3, 1, 2).float()
                c=torch.cat((c1,c2),dim=1)
            elif c_one_hot==0:
                c = self.embedding(c.view(-1)).view(shp)  # (B, H, W, C)
                c = c.permute(0, 3, 1, 2)
            if self.c_one_hot==3:
                x = torch.cat((x,c),dim=1)
                c=None
            else:
                c = self.cond_resnet(c.float())
        x_v, x_h = (x, x)
        for i, layer in enumerate(self.layers):
            x_v, x_h = layer(x_v, x_h, c=c)

        return self.out(x_h)

    def generate(self, label, c, shape=(8, 8), batch_size=64):
        param = next(self.parameters())
        x = torch.zeros(
            (batch_size, *shape),
            dtype=torch.int64, device=param.device
        )
        for i in range(shape[0]):
            for j in range(shape[1]):
                logits = self.forward(x, c)
                probs = F.softmax(logits[:, :, i, j], -1)
                x.data[:, i, j].copy_(
                    probs.multinomial(1).squeeze().data
                )
        return x

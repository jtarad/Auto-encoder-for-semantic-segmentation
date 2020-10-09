import torch
from fastai.callbacks import *
from fastai.vision import *
from torchsummary import summary

def conv_block(c_in, c_out, ks, num_groups=None, **conv_kwargs):
    "A sequence of modules composed of Group Norm, ReLU and Conv3d in order"
    if not num_groups : num_groups = int(c_in/2) if c_in%2 == 0 else None
    return nn.Sequential(nn.GroupNorm(num_groups, c_in),
                         nn.ReLU(),
                         nn.Conv2d(c_in, c_out, ks, **conv_kwargs))

def reslike_block(nf, num_groups=None, bottle_neck:bool=False, **conv_kwargs):
    "A ResNet-like block with the GroupNorm normalization providing optional bottle-neck functionality"
    nf_inner = nf / 2 if bottle_neck else nf
    return SequentialEx(conv_block(num_groups=num_groups, c_in=nf, c_out=nf_inner, ks=3, stride=1, padding=1, **conv_kwargs),
                        conv_block(num_groups=num_groups, c_in=nf_inner, c_out=nf, ks=3, stride=1, padding=1, **conv_kwargs),
                        MergeLayer())

def upsize(c_in, c_out, ks=1, scale=2):
    "Reduce the number of features by 2 using Conv with kernel size 1x1x1 and double the spatial dimension using 2D bilinear upsampling"
    return nn.Sequential(nn.Conv2d(c_in, c_out, ks),
                       nn.Upsample(scale_factor=scale, mode='bilinear', align_corners=True))
    #return nn.ConvTranspose2d(c_in, c_out, ks)

class Encoder(nn.Module):
    "Encoder part"
    def __init__(self, latent_dim:int=256):
        super().__init__()
        self.latent_dim = latent_dim

        self.conv1 = nn.Conv2d(110, 32, 3, stride=1, padding=1)         
        self.res_block1 = reslike_block(32, num_groups=8)
        self.conv_block1 = conv_block(32, 64, 3, num_groups=8, stride=2, padding=1)
        self.res_block2 = reslike_block(64, num_groups=8)
        self.conv_block2 = conv_block(64, 64, 3, num_groups=8, stride=1, padding=1)
        self.res_block3 = reslike_block(64, num_groups=8)
        self.conv_block3 = conv_block(64, 128, 3, num_groups=8, stride=2, padding=1)
        self.res_block4 = reslike_block(128, num_groups=8)
        self.conv_block4 = conv_block(128, 128, 3, num_groups=8, stride=1, padding=1)
        self.res_block5 = reslike_block(128, num_groups=8)
        self.conv_block5 = conv_block(128, 256, 3, num_groups=8, stride=2, padding=1)
        self.res_block6 = reslike_block(256, num_groups=8)
        self.conv_block6 = conv_block(256, 256, 3, num_groups=8, stride=1, padding=1)
        self.res_block7 = reslike_block(256, num_groups=8)
        self.conv_block7 = conv_block(256, 256, 3, num_groups=8, stride=1, padding=1)
        self.res_block8 = reslike_block(256, num_groups=8)
        self.conv_block8 = conv_block(256, latent_dim, 3, num_groups=8, stride=1, padding=1)
        self.res_block9 = reslike_block(latent_dim, num_groups=8)

    def forward(self, x):
        x = self.conv1(x)                                           # Output size: (1, 32, 64, 64)
        x = self.res_block1(x)                                      # Output size: (1, 32, 64, 64)
        x = self.conv_block1(x)                                     # Output size: (1, 64, 32, 32)
        x = self.res_block2(x)                                      # Output size: (1, 64, 32, 32)
        x = self.conv_block2(x)                                     # Output size: (1, 64, 32, 32)
        x = self.res_block3(x)                                      # Output size: (1, 64, 32, 32)
        x = self.conv_block3(x)                                     # Output size: (1, 128, 16, 16)
        x = self.res_block4(x)                                      # Output size: (1, 128, 16, 16)
        x = self.conv_block4(x)                                     # Output size: (1, 128, 16, 16)
        x = self.res_block5(x)                                      # Output size: (1, 128, 16, 16)
        x = self.conv_block5(x)                                     # Output size: (1, 256, 8, 8)
        x = self.res_block6(x)                                      # Output size: (1, 256, 8, 8)
        x = self.conv_block6(x)                                     # Output size: (1, 256, 8, 8)
        x = self.res_block7(x)                                      # Output size: (1, 256, 8, 8)
        x = self.conv_block7(x)                                     # Output size: (1, 256, 8, 8)
        x = self.res_block8(x)                                      # Output size: (1, 256, 8, 8)
        x = self.conv_block8(x)                                     # Output size: (1, 256, 8, 8)
        x = self.res_block9(x)                                      # Output size: (1, 256, 8, 8)
        return x

class Decoder(nn.Module):
    "Decoder Part"
    def __init__(self):
        super().__init__()
        self.upsize1 = upsize(256, 128)
        self.reslike1 = reslike_block(128, num_groups=8)
        self.upsize2 = upsize(128, 64)
        self.reslike2 = reslike_block(64, num_groups=8)
        self.upsize3 = upsize(64, 32)
        self.reslike3 = reslike_block(32, num_groups=8)
        self.conv1 = nn.Conv2d(32, 110, 1) 
        self.sigmoid1 = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.upsize1(x)                                         # Output size: (1, 128, 16, 16)
        x = x + hooks_ae.stored[2]                                  # Output size: (1, 128, 16, 16)
        x = self.reslike1(x)                                        # Output size: (1, 128, 16, 16)
        x = self.upsize2(x)                                         # Output size: (1, 64, 32, 32)
        x = x + hooks_ae.stored[1]                                  # Output size: (1, 64, 32, 32)
        x = self.reslike2(x)                                        # Output size: (1, 64, 32, 32)
        x = self.upsize3(x)                                         # Output size: (1, 32, 64, 64)
        x = x + hooks_ae.stored[0]                                  # Output size: (1, 32, 64, 64)
        x = self.reslike3(x)                                        # Output size: (1, 32, 64, 64)
        x = self.conv1(x)                                           # Output size: (1, 110, 64, 64)
        return x

class DecoderSeg(nn.Module):
    "Decoder Part"
    def __init__(self):
        super().__init__()
        self.upsize1 = upsize(256, 128)
        self.reslike1 = reslike_block(128, num_groups=8)
        self.upsize2 = upsize(128, 64)
        self.reslike2 = reslike_block(64, num_groups=8)
        self.upsize3 = upsize(64, 32)
        self.reslike3 = reslike_block(32, num_groups=8)
        self.convdec = nn.Conv2d(32, 110, 1) 
        self.convseg = nn.Conv2d(32, 15, 1)
        self.sigmoid1 = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.upsize1(x)                                         # Output size: (1, 128, 16, 16)
        x = x + hooks_seg.stored[2]                                 # Output size: (1, 128, 16, 16)
        x = self.reslike1(x)                                        # Output size: (1, 128, 16, 16)
        x = self.upsize2(x)                                         # Output size: (1, 64, 32, 32)
        x = x + hooks_seg.stored[1]                                 # Output size: (1, 64, 32, 32)
        x = self.reslike2(x)                                        # Output size: (1, 64, 32, 32)
        x = self.upsize3(x)                                         # Output size: (1, 32, 64, 64)
        x = x + hooks_seg.stored[0]                                 # Output size: (1, 32, 64, 64)
        x = self.reslike3(x)                                        # Output size: (1, 32, 64, 64)
        out_seg = self.convseg(x)                                   # Output size: (1, 15, 64, 64)
        return out_seg

class DecoderTwoHead(nn.Module):
    "Decoder Part"
    def __init__(self):
        super().__init__()
        self.upsize1 = upsize(256, 128)
        self.reslike1 = reslike_block(128, num_groups=8)
        self.upsize2 = upsize(128, 64)
        self.reslike2 = reslike_block(64, num_groups=8)
        self.upsize3 = upsize(64, 32)
        self.reslike3 = reslike_block(32, num_groups=8)
        self.convdec = nn.Conv2d(32, 110, 1) 
        self.convseg = nn.Conv2d(32, 15, 1)
        self.sigmoid1 = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.upsize1(x)                                         # Output size: (1, 128, 16, 16)
        x = x + hooks_2h.stored[2]                                 # Output size: (1, 128, 16, 16)
        x = self.reslike1(x)                                        # Output size: (1, 128, 16, 16)
        x = self.upsize2(x)                                         # Output size: (1, 64, 32, 32)
        x = x + hooks_2h.stored[1]                                 # Output size: (1, 64, 32, 32)
        x = self.reslike2(x)                                        # Output size: (1, 64, 32, 32)
        x = self.upsize3(x)                                         # Output size: (1, 32, 64, 64)
        x = x + hooks_2h.stored[0]                                 # Output size: (1, 32, 64, 64)
        x = self.reslike3(x)                                        # Output size: (1, 32, 64, 64)
        out_dec = self.convdec(x)                                   # Output size: (1, 110, 64, 64)
        out_seg = self.convseg(x)                                   # Output size: (1, 15, 64, 64)
        return out_dec, out_seg


class Autoencoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder = Encoder(latent_dim=256)
    self.decoder = Decoder()

  def forward(self, input):
    interm_res = self.encoder(input)
    top_res = self.decoder(interm_res)                               # Output size: (1, 110, 64, 64)
    return top_res

class AutoEncoderTwoHeads(nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder = Encoder(latent_dim=256)
    self.decoder = DecoderTwoHead()

  def forward(self, input):
    interm_res = self.encoder(input)
    res_dec, res_seg = self.decoder(interm_res)                      # Output size: (1, 110, 64, 64), (1, 15, 64, 64)
    return res_dec, res_seg

class AutoSeg(nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder = Encoder(latent_dim=256)
    self.decoder = DecoderSeg()

  def forward(self, input):
    interm_res = self.encoder(input)
    res_seg = self.decoder(interm_res)                                # Output size: (1, 15, 64, 64)
    return res_seg

autoencoder = Autoencoder().cuda()
autoencoder2heads = AutoencoderTwhoHeads().cuda()
autoseg = AutoSeg().cuda()


ms_seg = [autoseg.encoder.res_block1, 
          autoseg.encoder.res_block3, 
          autoseg.encoder.res_block5]
hooks_seg = hook_outputs(ms_seg, detach=False, grad=False)

ms_ae = [autoencoder.encoder.res_block1, 
         autoencoder.encoder.res_block3, 
         autoencoder.encoder.res_block5]
hooks_ae = hook_outputs(ms_ae, detach=False, grad=False)

ms_2h = [autoencoder2heads.encoder.res_block1, 
         autoencoder2heads.encoder.res_block3, 
         autoencoder2heads.encoder.res_block5]
hooks_2h = hook_outputs(ms_2h, detach=False, grad=False)

lr = 1e-4
optimizer_ae = optim.Adam(autoencoder.parameters(), lr)
optimizer_seg = optim.Adam(autoseg.parameters(), lr)
optimizer_2h = optim.Adam(autoencoder2heads.parameters(), lr)
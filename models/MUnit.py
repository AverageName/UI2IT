import sys
sys.path.append('..')
import torch
import os
import torch.nn as nn
import torchvision
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import random
from datasets.UnalignedDataset import UnalignedDataset
from utils.utils import *
from layers.ConvNormRelu import ConvNormRelu
from layers.ResBlock import ResBlock
from layers.LinearNormAct import LinearNormAct




class ContentEncoder(nn.Module):
    
    def __init__(self, num_channels, num_blocks, norm_type="instance", pad_type="reflection"):
        super(ContentEncoder, self).__init__()
        
        self.conv1 = ConvNormRelu(in_channels=num_channels, out_channels=64,
                                  kernel_size=7, padding=(3, pad_type), leaky=False, norm=norm_type)
        self.conv2 = ConvNormRelu(in_channels=64, out_channels=128, 
                                  kernel_size=4, padding=(1, pad_type), stride=2, leaky=False, norm=norm_type)
        self.conv3 = ConvNormRelu(in_channels=128, out_channels=256, 
                                 kernel_size=4, padding=(1, pad_type), stride=2, leaky=False, norm=norm_type)
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(ResBlock(in_planes=256, kernel_size=3, padding=(1, pad_type), norm=norm_type))
    
    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.conv3(out)
        for block in self.blocks:
            out = block(out)
        return out


class StyleEncoder(nn.Module):
    
    def __init__(self, num_channels, style_dims, norm_type="none", pad_type="reflection"):
        super(StyleEncoder, self).__init__()
        
        self.conv1 = ConvNormRelu(in_channels=num_channels, out_channels=64,
                                  kernel_size=7, padding=(3, pad_type), leaky=False, norm=norm_type)
        
        self.convs = nn.ModuleList()
        dims = 64
        prev_dims = 0
        n_convs = 4
        
        for _ in range(n_convs):
            prev_dims = dims
            dims = min(dims * 2, 256)
            
            self.convs.append(ConvNormRelu(in_channels=prev_dims, out_channels=dims, 
                                  kernel_size=4, padding=(1, pad_type), stride=2, leaky=False, norm=norm_type))
        
        self.conv_fc = nn.Conv2d(dims, style_dims, kernel_size=1, stride=1, padding=0)  
            
    def forward(self, inputs):
        out = self.conv1(inputs)
        for conv in self.convs:
            out = conv(out)
            
        #Fastest version of Global Average Pooling
        #out = torch.mean(out.view(out.size(0), out.size(1), -1), dim=2)
        #out = out.view(out.size(0), out.size(1), 1, 1)
        out = F.adaptive_avg_pool2d(out, 1)
        out = self.conv_fc(out)
        return out.view(out.size(0), -1)


class Decoder(nn.Module):
    
    def __init__(self, in_channels, norm_type="adain", pad_type="reflection"):
        super(Decoder, self).__init__()
        
        self.blocks = nn.ModuleList()
        n_blocks = 4
        for _ in range(n_blocks):
            self.blocks.append(ResBlock(in_planes=in_channels, kernel_size=3,
                                        padding=(1, pad_type), norm=norm_type))
        n_blocks = 2
        self.upsample_blocks = nn.ModuleList()
        prev_dims = 0
        dims = 256
        for _ in range(n_blocks):
            prev_dims = dims
            dims = dims // 2
            self.upsample_blocks.append(nn.Upsample(scale_factor=2))
            self.upsample_blocks.append(ConvNormRelu(in_channels=prev_dims, out_channels=dims,
                                                     kernel_size=5, padding=(2, pad_type), stride=1, norm="ln"))
            
        #self.last_layer = ConvNormRelu(in_channels=dims, out_channels=3, kernel_size=7,
                                      #padding=(3, pad_type), stride=1, norm=None)
        self.last_layer = nn.Conv2d(in_channels=dims, out_channels=3, kernel_size=7, padding=3, stride=1)
        
    def forward(self, inputs):
        out = inputs
        for block in self.blocks:
            out = block(out)
        for block in self.upsample_blocks:
            out = block(out)
        return F.tanh(self.last_layer(out))


class MLP(nn.Module):
    
    def __init__(self, in_channels, out_channels, hidden_dim, num_blocks, norm_type="none"):
        super(MLP, self).__init__()
        
        self.fc1 = LinearNormAct(in_channels=in_channels, out_channels=hidden_dim, norm=norm_type)
        
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks - 2):
            self.blocks.append(LinearNormAct(in_channels=hidden_dim, out_channels=hidden_dim, norm=norm_type))
        
        self.last_fc = LinearNormAct(in_channels=hidden_dim, out_channels=out_channels, norm=norm_type, activation="none")
        
    def forward(self, inputs):
        
        out = self.fc1(inputs)
        for block in self.blocks:
            out = block(out)
        return self.last_fc(out)


class MUnitAutoencoder(nn.Module):
    
    def __init__(self, in_channels, mlp_hidden_dim, mlp_num_blocks, enc_style_dims, enc_cont_num_blocks,
                 norm_type_cont, pad_type_cont, norm_type_style, pad_type_style, norm_type_decoder, pad_type_decoder,
                 norm_type_mlp, enc_cont_dim=256):
        super(MUnitAutoencoder, self).__init__()
        
        
        self.enc_cont = ContentEncoder(num_channels=in_channels, num_blocks=enc_cont_num_blocks,
                                       norm_type=norm_type_cont, pad_type=pad_type_cont)
        self.enc_style = StyleEncoder(num_channels=in_channels, style_dims=enc_style_dims, norm_type=norm_type_style,
                                      pad_type=pad_type_style)
        
        self.decoder = Decoder(in_channels=enc_cont_dim, norm_type=norm_type_decoder, pad_type=pad_type_decoder)
        self.mlp = MLP(in_channels=enc_style_dims, out_channels=self.get_num_adain_params(self.decoder),
                       hidden_dim=mlp_hidden_dim, num_blocks=mlp_num_blocks, norm_type=norm_type_mlp)
        
    
    
    
    def encode(self, inputs):
        enc_cont = self.enc_cont(inputs)
        enc_style = self.enc_style(inputs)
        return enc_cont, enc_style
    
    def decode(self, enc_cont, enc_style):
        features = self.mlp(enc_style)
        self.assign_adain_params(features, self.decoder)
        return self.decoder(enc_cont)
    
    def forward(self, inputs):
        enc_cont, enc_style = self.encode(inputs)
        rec_inputs = self.decode(enc_cont, enc_style)
        return rec_inputs

    def assign_adain_params(self, adain_params, model):
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                mean = adain_params[:, :m.num_features]
                std = adain_params[:, m.num_features:2*m.num_features]
                m.bias = mean.contiguous().view(-1)
                m.weight = std.contiguous().view(-1)
                if adain_params.size(1) > 2*m.num_features:
                    adain_params = adain_params[:, 2*m.num_features:]
    

    def get_num_adain_params(self, model):
        num_adain_params = 0
        for m in model.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                num_adain_params += 2*m.num_features
        return num_adain_params



class MSDiscriminator(nn.Module):
    
    def __init__(self, in_channels, num_scales):
        super(MSDiscriminator, self).__init__()
        
        self.discrs = nn.ModuleList()
        self.in_channels = in_channels
        self.num_scales = num_scales
        for _ in range(self.num_scales):
            self.discrs.append(self.create_discr(self.in_channels))
        
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, count_include_pad=False)
        
        
    
    def create_discr(self, in_channels):
        prev_dims = 0
        dims = 64
        self.discr = []
        n_blocks = 3
        
        self.discr += [ConvNormRelu(in_channels=in_channels, out_channels=dims, 
                                  kernel_size=4, padding=(1, "reflection"), stride=2, leaky=True, norm='none')]
        
        for _ in range(n_blocks):
            prev_dims = dims
            dims = dims * 2
            self.discr += [ConvNormRelu(in_channels=prev_dims, out_channels=dims, 
                                  kernel_size=4, padding=(1, "reflection"), stride=2, leaky=True, norm='none')]
        
        self.discr += [nn.Conv2d(dims, out_channels=1, kernel_size=1, padding=0)]
        return nn.Sequential(*self.discr)
        
    
    def forward(self, inputs):
        outputs = []
        for discr in self.discrs:
            outputs.append(discr(inputs))
            inputs = self.downsample(inputs)
        return outputs
    
    def discr_loss(self, real, fake):
        outputs_real = self.forward(real)
        outputs_fake = self.forward(fake.detach())
        loss_fake = 0
        loss_real = 0
        #print(outputs_real)
        #print(outputs_fake)
        for i in range(self.num_scales):
            loss_fake += calc_mse_loss(outputs_fake[i], 0.0)
            loss_real += calc_mse_loss(outputs_real[i], 1.0)
        
        #print("Loss_D Fake: ", loss_fake)
        #print("Loss_D Real: ", loss_real)
        loss = loss_fake + loss_real
        #print("One of discriminators loss: ", loss)
        
        return loss
    
    
    def gen_loss(self, fake):
        loss = 0
        outputs_fake = self.forward(fake)
        for i in range(self.num_scales):
            loss += calc_mse_loss(outputs_fake[i], 1.0) 
        return loss


class MUnit_pytorch(nn.Module):
    
    
    def __init__(self, in_channels, mlp_hidden_dim, mlp_num_blocks, d_num_scales, enc_style_dims,
                 enc_cont_num_blocks, norm_type_cont, pad_type_cont, norm_type_style, pad_type_style, norm_type_decoder, pad_type_decoder,
                 norm_type_mlp, enc_cont_dim=256, use_perceptual_loss=False):
              
                 
        super(MUnit_pytorch, self).__init__()
        
        
        self.G1 = MUnitAutoencoder(in_channels=in_channels, mlp_hidden_dim=mlp_hidden_dim, mlp_num_blocks=mlp_num_blocks,
                                   enc_style_dims=enc_style_dims, enc_cont_num_blocks=enc_cont_num_blocks, norm_type_cont=norm_type_cont,
                                   pad_type_cont=pad_type_cont, norm_type_style=norm_type_style, pad_type_style=pad_type_style,
                                   norm_type_decoder=norm_type_decoder, pad_type_decoder=pad_type_decoder, norm_type_mlp=norm_type_mlp,
                                   enc_cont_dim=enc_cont_dim)
        
        self.G2 = MUnitAutoencoder(in_channels=in_channels, mlp_hidden_dim=mlp_hidden_dim, mlp_num_blocks=mlp_num_blocks,
                                   enc_style_dims=enc_style_dims, enc_cont_num_blocks=enc_cont_num_blocks, norm_type_cont=norm_type_cont,
                                   pad_type_cont=pad_type_cont, norm_type_style=norm_type_style, pad_type_style=pad_type_style,
                                   norm_type_decoder=norm_type_decoder, pad_type_decoder=pad_type_decoder, norm_type_mlp=norm_type_mlp,
                                   enc_cont_dim=enc_cont_dim)
        
        self.D1 = MSDiscriminator(in_channels, num_scales=d_num_scales)
        
        self.D2 = MSDiscriminator(in_channels, num_scales=d_num_scales)
        

        self.use_perceptual_loss = use_perceptual_loss

        if use_perceptual_loss:
            self.vgg = load_vgg_feature_extractor()

    
    def forward(self, domain_A, domain_B):
        
        cont_A, style_A = self.G1.encode(domain_A)
        cont_B, style_B = self.G2.encode(domain_B)
        
        fake_style_B = torch.randn(*style_B.shape).cuda()
        fake_style_A = torch.randn(*style_A.shape).cuda()
        
        fake_A = self.G1.decode(cont_B, fake_style_A)
        fake_B = self.G2.decode(cont_A, fake_style_B)
        
        return fake_A, fake_B
        
    def backward_Gs(self, domain_A, domain_B):
        
        
        cont_A, style_A = self.G1.encode(domain_A)
        cont_B, style_B = self.G2.encode(domain_B)
        
        
        fake_style_B = torch.randn((*style_B.shape), requires_grad=True).cuda()
        fake_style_A = torch.randn((*style_A.shape), requires_grad=True).cuda()
        
        fake_A = self.G1.decode(cont_B, fake_style_A)
        fake_B = self.G2.decode(cont_A, fake_style_B)
        
        #Reconstructed images
        rec_img_A = self.G1.decode(cont_A, style_A)
        rec_img_B = self.G2.decode(cont_B, style_B)
        
        
        #Reconstructed latent contents and styles
        rec_cont_A, rec_fake_style_B = self.G2.encode(fake_B)
        rec_cont_B, rec_fake_style_A = self.G1.encode(fake_A)
        
        #Loss of reconstructed images
        img_rec_loss_A = F.l1_loss(rec_img_A, domain_A)
        img_rec_loss_B = F.l1_loss(rec_img_B, domain_B)
        #print("img_rec_loss: ", img_rec_loss_A, img_rec_loss_B)
        
        #Loss of reconstructed latent content and styles
        cont_loss_A = torch.mean(torch.abs(rec_cont_A - cont_A))
        cont_loss_B = torch.mean(torch.abs(rec_cont_B - cont_B))
        #print("cont_loss: ", cont_loss_A, cont_loss_B)
        
        style_loss_A = torch.mean(torch.abs(rec_fake_style_A - fake_style_A))
        style_loss_B = torch.mean(torch.abs(rec_fake_style_B - fake_style_B))
        #print("style_loss: ", style_loss_A, style_loss_B)
        
        #Adversarial loss of generated pics
        adv_loss_A = self.D2.gen_loss(fake_B)
        adv_loss_B = self.D1.gen_loss(fake_A)
        #print("Adv loss: ", adv_loss_A, adv_loss_B)
    

        loss = 10 * (img_rec_loss_A + img_rec_loss_B) + (adv_loss_A + adv_loss_B) + \
               (cont_loss_A + cont_loss_B) + (style_loss_A + style_loss_B)

        #Perceptual loss
        if self.use_perceptual_loss:
            perceptual_loss_A = calc_IN_feature_distance(self.vgg, [domain_A, fake_B])
            perceptual_loss_B = calc_IN_feature_distance(self.vgg, [domain_B, fake_A])
            loss = loss + perceptual_loss_A + perceptual_loss_B
            print("PERCEPTUAL: ", perceptual_loss_B, perceptual_loss_A)
        
        #print("Loss: ", loss)
        #loss.backward()
        
        return loss
    
    def backward_Ds(self, real_A, fake_A, real_B, fake_B):
        loss_D1 = self.D1.discr_loss(real_A, fake_A)
        loss_D2 = self.D2.discr_loss(real_B, fake_B)
        #print("loss D1 and D2: ", loss_D1, loss_D2)
        loss = loss_D1 + loss_D2
        #print("loss D: ", loss)
        #loss.backward()
        
        return loss
















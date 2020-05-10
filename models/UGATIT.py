import sys
sys.path.append('..')
#import os
#print(os.path.abspath(os.getcwd()))
import torch
import torch.nn as nn
from layers.ConvNormRelu import *
from layers.AdaLINResBlock import *
from layers.ConvSpectralNormAct import *
from layers.ResBlock import *
from layers.UpsampleConvBlock import *
from utils.utils import RhoClipper, calc_mse_loss




class Generator(nn.Module):
    
    def __init__(self, in_channels, img_size, num_enc_blocks, num_enc_res_blocks, num_dec_upsample_blocks,
                 num_dec_res_blocks, norm_type, pad_type):
        super(Generator, self).__init__()
        
        dims = 64
        self.conv1 = ConvNormRelu(in_channels=in_channels, out_channels=dims, kernel_size=7, padding=(3, pad_type),
                                  norm=norm_type, leaky=False)
        
       # self.conv1_5 = ConvNormRelu(in_channels=64, out_channels=dims, kernel_size=7, padding=(3, pad_type),
        #                           norm=norm_type, leaky=False)
        
        self.convs = nn.ModuleList()
        
        for _ in range(num_enc_blocks - 1):
            prev_dims = dims
            dims = min(dims * 2, 256)
            self.convs.append(ConvNormRelu(in_channels=prev_dims, out_channels=dims, kernel_size=3,
                                           padding=(1, pad_type), norm=norm_type, stride=2))
            
        self.res_blocks = nn.ModuleList()
        
        for _ in range(num_enc_res_blocks):
            self.res_blocks.append(ResBlock(in_planes=dims, kernel_size=3, padding=(1, pad_type), norm=norm_type))
            
        
        self.gap_fc = nn.Linear(dims, 1)
        self.gmp_fc = nn.Linear(dims, 1)
        self.conv2 = nn.Conv2d(in_channels=dims * 2, out_channels=dims, kernel_size=1, stride=1)

        self.mlp_downsample = ConvNormRelu(in_channels=dims, out_channels=dims, kernel_size=3,
                                           padding=(1, pad_type), norm=norm_type, stride=2)
        
        num_mlp_downsamples = 1
        
        MLP = [nn.Linear(in_features=dims * (img_size // 2 ** (num_enc_blocks - 1 + num_mlp_downsamples)) ** 2, out_features=dims),
               nn.ReLU(True),
               nn.Linear(in_features=dims, out_features=dims),
               nn.ReLU(True)]
        
        # MLP = [nn.Linear(in_features=dims, out_features=dims),
        #        nn.ReLU(True),
        #        nn.Linear(in_features=dims, out_features=dims),
        #        nn.ReLU(True)]
        
        self.mlp = nn.Sequential(*MLP)
        self.gamma = nn.Linear(in_features=dims, out_features=dims)
        self.beta = nn.Linear(in_features=dims, out_features=dims)
        
        #Decoder
        self.decoder_res_blocks = nn.ModuleList()
        for _ in range(num_dec_res_blocks):
            self.decoder_res_blocks.append(AdaLINResBlock(in_channels=dims, kernel_size=3,
                                                      activation='relu', pad_type="reflection"))
        
        self.upsample_blocks = nn.ModuleList()
        for _ in range(num_dec_upsample_blocks):
            self.upsample_blocks.append(UpsampleConvBlock(in_channels=dims, kernel_size=3, activation="relu",
                                                         norm_type="lin", pad_type="reflection"))
            dims = dims // 2
        
        self.pad_last_conv = nn.ReflectionPad2d(3)
        self.last_conv = nn.Conv2d(in_channels=dims, out_channels=3, kernel_size=7, stride=1)
        
    
    def forward(self, inputs):
        #print(list(self.gap_fc.parameters())[0].shape)
        #Extracting features
        out = self.conv1(inputs)
        #out = self.conv1_5(out)
        
        for conv in self.convs:
            out = conv(out)
            
        for res_block in self.res_blocks:
            out = res_block(out)
            
            
        #Global Average and Max Pooling
        gap = F.adaptive_avg_pool2d(out, 1)
        gap_logits = self.gap_fc(gap.view(out.shape[0], -1))
        gap_fc_weigths = list(self.gap_fc.parameters())[0]
        gap = gap_fc_weigths.view(gap_fc_weigths.size(0), gap_fc_weigths.size(1), 1, 1) * out
        
        
        gmp = F.adaptive_max_pool2d(out, 1)
        gmp_logits = self.gmp_fc(gmp.view(out.shape[0], -1))
        gmp_fc_weights = list(self.gmp_fc.parameters())[0]
        gmp = gmp_fc_weights.view(gmp_fc_weights.size(0), gmp_fc_weights.size(1), 1, 1) * out
        
        gmp_gap_logits = torch.cat([gmp_logits, gap_logits], 1)
        
        gmp_gap = torch.cat([gmp, gap], 1)
        
        out = F.relu(self.conv2(gmp_gap))
        cam = out
        
        #Calculating beta and gamma for AdaLIN
        #out = F.adaptive_avg_pool2d(out, 1)
        out = self.mlp_downsample(out)
        out = self.mlp(out.view(out.size(0), -1))
        
        gamma = self.gamma(out)
        beta = self.beta(out)
        
        out = cam
        for res_block in self.decoder_res_blocks:
            out = res_block(out, gamma, beta)
        
        for upsample_block in self.upsample_blocks:
            out = upsample_block(out)
        
        out = self.pad_last_conv(out)
        return F.tanh(self.last_conv(out)), gmp_gap_logits


class PatchGan(nn.Module):
    
    def __init__(self, in_channels, num_downsample, pad_type):
        super(PatchGan, self).__init__()
        
        dims = 64
        self.conv1 = ConvSpectralNormAct(in_channels=in_channels, out_channels=dims, kernel_size=4,
                                  padding=(1, pad_type), stride=2, activation="lrelu")
        
        #self.conv1_5 = ConvSpectralNormAct(in_channels=64, out_channels=dims, kernel_size=4,
                           #       padding=(1, pad_type), stride=2, activation="lrelu")
        
        self.convs = nn.ModuleList()
        for _ in range(num_downsample - 1):
            prev_dims = dims
            dims = dims * 2
            self.convs.append(ConvSpectralNormAct(in_channels=prev_dims, out_channels=dims, kernel_size=4,
                                  padding=(1, pad_type), stride=2, activation="lrelu"))
        
        prev_dims = dims
        dims = dims * 2
        
        self.conv2 = ConvSpectralNormAct(in_channels=prev_dims, out_channels=dims, kernel_size=4,
                                  padding=(1, pad_type), stride=1, activation="lrelu")
        
        self.gap_fc = nn.utils.spectral_norm(nn.Linear(dims, 1, bias=False))
        self.gmp_fc = nn.utils.spectral_norm(nn.Linear(dims, 1, bias=False))
        self.conv3 = nn.Conv2d(in_channels=dims * 2, out_channels=dims, kernel_size=1, stride=1, padding=0)
        
        self.pad_last_conv = nn.ReflectionPad2d(1)
        self.last_conv = nn.utils.spectral_norm(nn.Conv2d(in_channels=dims, out_channels=1, kernel_size=4, stride=1))
    
    
    def forward(self, inputs):
        #print(list(self.gap_fc.parameters())[0].shape)
        out = self.conv1(inputs)
       # out = self.conv1_5(out)
        for conv in self.convs:
            out = conv(out)
        out = self.conv2(out)
        
        gap = F.adaptive_avg_pool2d(out, 1)
        gap_logits = self.gap_fc(gap.view(gap.size(0), -1))
        gap_fc_weights = list(self.gap_fc.parameters())[0]
        #print(gap_fc_weights.shape)
        gap = out * gap_fc_weights.view(gap_fc_weights.size(0), gap_fc_weights.size(1), 1, 1)
        

        gmp = F.adaptive_max_pool2d(out, 1)
        gmp_logits = self.gmp_fc(gmp.view(gmp.size(0), -1))
        gmp_fc_weights = list(self.gmp_fc.parameters())[0]
        gmp = out * gmp_fc_weights.view(gmp_fc_weights.size(0), gmp_fc_weights.size(1), 1, 1)
        
        
        gap_gmp_logits = torch.cat([gap_logits, gmp_logits], 1)
        gap_gmp = torch.cat([gap, gmp], 1)
        
        out = F.leaky_relu(self.conv3(gap_gmp), negative_slope=0.2)
        
        out = self.pad_last_conv(out)
        out = self.last_conv(out)
        
        return out, gap_gmp_logits


class UGATIT_pytorch(nn.Module):
    
    
    def __init__(self, in_channels, img_size, num_enc_blocks, num_enc_res_blocks, num_dec_upsample_blocks,
                 num_dec_res_blocks, norm_type, pad_type, local_discr_num_downsample, global_discr_num_downsample):
        super(UGATIT_pytorch, self).__init__()
        
        self.G_AB = Generator(in_channels, img_size, num_enc_blocks, num_enc_res_blocks,
                            num_dec_upsample_blocks, num_dec_res_blocks, norm_type, pad_type)
        
        self.G_BA = Generator(in_channels, img_size, num_enc_blocks, num_enc_res_blocks,
                            num_dec_upsample_blocks, num_dec_res_blocks, norm_type, pad_type)
        
        self.D_AG = PatchGan(in_channels, num_downsample=global_discr_num_downsample, pad_type=pad_type)
        self.D_AL = PatchGan(in_channels, num_downsample=local_discr_num_downsample, pad_type=pad_type)
        
        self.D_BG = PatchGan(in_channels, num_downsample=global_discr_num_downsample, pad_type=pad_type)
        self.D_BL = PatchGan(in_channels, num_downsample=local_discr_num_downsample, pad_type=pad_type)
        self.rho_clipper = RhoClipper(0, 1)
        
    
    def forward(self, domain_A, domain_B):
        
        fake_B, logits_cam_AB = self.G_AB(domain_A)
        fake_A, logits_cam_BA = self.G_BA(domain_B)
        
        return fake_A, fake_B
    
    def backward_Gs(self, domain_A, domain_B):
        
        fake_B, logits_cam_AB = self.G_AB(domain_A)
        fake_A, logits_cam_BA = self.G_BA(domain_B)
        
        D_AG_out_fake, D_AG_logits_fake = self.D_AG(fake_A)
        D_AL_out_fake, D_AL_logits_fake = self.D_AL(fake_A)
        
        D_AG_out_real, D_AG_logits_real = self.D_AG(domain_A)
        D_AL_out_real, D_AL_logits_real = self.D_AL(domain_A)
        
        D_BG_out_fake, D_BG_logits_fake = self.D_BG(fake_B)
        D_BL_out_fake, D_BL_logits_fake = self.D_BL(fake_B)
        
        D_BG_out_real, D_BG_logits_real = self.D_BG(domain_B)
        D_BL_out_real, D_BL_logits_real = self.D_BL(domain_B)
        
        
        rec_A, _ = self.G_BA(fake_B)
        rec_B, _ = self.G_AB(fake_A)
        
        id_A, logits_cam_AA = self.G_BA(domain_A)
        id_B, logits_cam_BB = self.G_AB(domain_B)
        
        
        #Adversarial loss for fake_A
        adv_loss_A = (calc_mse_loss(D_AG_out_fake, 1.0) + calc_mse_loss(D_AL_out_fake, 1.0) + \
                     calc_mse_loss(D_AG_out_real, 0.0) + calc_mse_loss(D_AL_out_real, 0.0))
        
        #Adversarial loss for fake_B
        adv_loss_B = (calc_mse_loss(D_BG_out_fake, 1.0) + calc_mse_loss(D_BL_out_fake, 1.0) + \
                     calc_mse_loss(D_BG_out_real, 0.0) + calc_mse_loss(D_BL_out_real, 0.0))
        
        #Cycle loss for domain_A and domain_B
        
        cycle_loss_ABA = F.l1_loss(rec_A, domain_A)
        cycle_loss_BAB = F.l1_loss(rec_B, domain_B)
        
        #Identity loss
        
        identity_loss_A = F.l1_loss(id_A, domain_A)
        
        
        identity_loss_B = F.l1_loss(id_B, domain_B)
        
        #CAM Loss
        cam_loss_AB = F.binary_cross_entropy_with_logits(logits_cam_AB, torch.ones_like(logits_cam_AB).cuda()) + \
                        F.binary_cross_entropy_with_logits(logits_cam_BB, torch.zeros_like(logits_cam_BB).cuda())
        
        cam_loss_BA = F.binary_cross_entropy_with_logits(logits_cam_BA, torch.ones_like(logits_cam_BA).cuda()) + \
                        F.binary_cross_entropy_with_logits(logits_cam_AA, torch.zeros_like(logits_cam_AA).cuda())
        
        cam_d_loss_A = calc_mse_loss(D_AG_logits_fake, 1.0) + calc_mse_loss(D_AL_logits_fake, 1.0)
        
        cam_d_loss_B = calc_mse_loss(D_BG_logits_fake, 1.0) + calc_mse_loss(D_BL_logits_fake, 1.0)
        
        loss = 1000 * (cam_loss_AB + cam_loss_BA) + 10 * (identity_loss_A + identity_loss_B) + \
                10 * (cycle_loss_ABA + cycle_loss_BAB) + (adv_loss_A + adv_loss_B + cam_d_loss_A + cam_d_loss_B)
        
        #loss.backward()
        
        return loss
    
    
    def backward_Ds(self, domain_A, domain_B):
        
        fake_B, logits_cam_AB = self.G_AB(domain_A)
        fake_A, logits_cam_BA = self.G_BA(domain_B)
        
        D_AG_out_fake, D_AG_logits_fake = self.D_AG(fake_A.detach())
        D_AL_out_fake, D_AL_logits_fake = self.D_AL(fake_A.detach())
        
        D_AG_out_real, D_AG_logits_real = self.D_AG(domain_A)
        D_AL_out_real, D_AL_logits_real = self.D_AL(domain_A)
        
        D_BG_out_fake, D_BG_logits_fake = self.D_BG(fake_B.detach())
        D_BL_out_fake, D_BL_logits_fake = self.D_BL(fake_B.detach())
        
        D_BG_out_real, D_BG_logits_real = self.D_BG(domain_B)
        D_BL_out_real, D_BL_logits_real = self.D_BL(domain_B)
        
        cam_d_loss_A = calc_mse_loss(D_AG_logits_fake, 0.0) + calc_mse_loss(D_AL_logits_fake, 0.0) + \
                        calc_mse_loss(D_AG_logits_real, 1.0) + calc_mse_loss(D_AL_logits_real, 1.0)
        
        cam_d_loss_B = calc_mse_loss(D_BG_logits_fake, 0.0) + calc_mse_loss(D_BL_logits_fake, 0.0) + \
                        calc_mse_loss(D_BG_logits_real, 1.0) + calc_mse_loss(D_BL_logits_real, 1.0)
        
        adv_loss_d_A = calc_mse_loss(D_AG_out_fake, 0.0) + calc_mse_loss(D_AL_out_fake, 0.0) + \
                        calc_mse_loss(D_AG_out_real, 1.0) + calc_mse_loss(D_AL_out_real, 1.0)
        
        adv_loss_d_B = calc_mse_loss(D_BG_out_fake, 0.0) + calc_mse_loss(D_BL_out_fake, 0.0) + \
                        calc_mse_loss(D_BG_out_real, 1.0) + calc_mse_loss(D_BL_out_real, 1.0)
        
        loss = cam_d_loss_A + cam_d_loss_B + adv_loss_d_A + adv_loss_d_B

        #loss.backward()
        
        return loss
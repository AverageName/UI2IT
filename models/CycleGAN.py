import sys
sys.path.append('..')
import torch
import torch.nn as nn
from layers.ConvNormRelu import ConvNormRelu
from layers.ResBlock import ResBlock
from utils.utils import calc_mse_loss
import torch.nn.functional as F

#Figure out real PatchGan
class PatchGan(nn.Module):
    
    def __init__(self, input_channels, norm_type):
        super(PatchGan, self).__init__()
        
        self.layer1 = ConvNormRelu(in_channels=input_channels, out_channels=64, kernel_size=4,
                                        padding=(1, "zeros"), stride=2, norm=None)
        self.layer2 = ConvNormRelu(in_channels=64, out_channels=128, kernel_size=4,
                                        padding=(1, "zeros"), stride=2, norm=norm_type)
        self.layer3 = ConvNormRelu(in_channels=128, out_channels=256, kernel_size=4,
                                        padding=(1, "zeros"), stride=2, norm=norm_type)
        self.layer4 = ConvNormRelu(in_channels=256, out_channels=512, kernel_size=4,
                                        padding=(1, "zeros"), stride=1, norm=norm_type)
        
        self.conv_fc = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4,
                                 padding=1, stride=1)
    
    def forward(self, inputs):
        out = self.layer1(inputs)
        #print(out.shape)
        out = self.layer2(out)
        #print(out.shape)
        out = self.layer3(out)
        #print(out.shape)
        out = self.layer4(out)
        #print(out.shape)
        out = self.conv_fc(out)
        #print(out.shape)
        return out
    
class ResnetGenerator(nn.Module):
    
    def __init__(self, in_channels, n_blocks, norm_type='batch'):
        super(ResnetGenerator, self).__init__()
        
        self.conv1 = ConvNormRelu(in_channels=in_channels, out_channels=64, kernel_size=7,
                                       padding=(3, "reflection"), stride=1, norm=norm_type, leaky=False)
        self.conv2 = ConvNormRelu(in_channels=64, out_channels=128, kernel_size=3,
                                  padding=(1, "zeros"), stride=2, norm=norm_type, leaky=False)
        self.conv3 = ConvNormRelu(in_channels=128, out_channels=256, kernel_size=3,
                                  padding=(1, "zeros"), stride=2, norm=norm_type, leaky=False)
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            self.blocks.append(ConvNormRelu(in_channels=256, out_channels=256, kernel_size=3,
                                            padding=(1, "reflection"), stride=1, norm=norm_type, leaky=False))
        self.conv4 = ConvNormRelu(in_channels=256, out_channels=128, kernel_size=3, 
                                  padding=(1, "zeros"), stride=2, norm=norm_type, leaky=False, conv_type="transpose")
        self.conv5 = ConvNormRelu(in_channels=128, out_channels=64, kernel_size=3, 
                                  padding=(1, "zeros"), stride=2, norm=norm_type, leaky=False, conv_type="transpose")
        
        self.pad = nn.ReflectionPad2d(3)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=7, padding=0)
        
    def forward(self, inputs):
        out = self.conv1(inputs)
        #print(out.shape)
        out = self.conv2(out)
        #print(out.shape)
        out = self.conv3(out)
        #print(out.shape)
        for block in self.blocks:
            out = block(out)
            #print(out.shape)
        out = self.conv4(out)
        #print(out.shape)
        out = self.conv5(out)
        #print(out.shape)
        out = self.pad(out)
        out = self.conv6(out)
        #print(out.shape)
        return F.tanh(out)

# class CycleGAN_pytorch(nn.Module):
    
#     def __init__(self, in_channels, n_blocks, norm_type_gen, norm_type_discr):
#         self.G1 = ResnetGenerator(in_channles, n_blocks, norm_type_gen)
#         self.G2 = ResnetGenerator(in_channles, n_blocks, norm_type_gen)
#         self.D1 = PatchGAN(in_channels, norm_type_discr)
#         self.D2 = PatchGAN(in_channels, norm_type_discr)
    
    
def calc_Gs_outputs(G1, G2, real_A, real_B):
    fake_B = G1(real_A)
    cycle_BA = G2(fake_B)
    fake_A = G2(real_B)
    cycle_AB = G1(fake_A)
    return fake_B, cycle_BA, fake_A, cycle_AB

def backward_D(real, fake, D):
    real_output = D(real)
    #print(real_output.shape)
    #print(real_output)
    d_real_loss = calc_mse_loss(real_output, 1.0)
    #F.mse_loss(real_output, torch.ones(real_output.shape).cuda())
    
    fake_output = D(fake.detach())
    d_fake_loss = F.mse_loss(fake_output, torch.zeros(fake_output.shape).cuda())
    
    loss = (d_fake_loss + d_real_loss) * 0.5
    #print("Discr loss: ", loss)
    loss.backward()
    return loss

    
def backward_Gs(fake_B, cycle_BA, fake_A, cycle_AB, real_A, real_B, G1, G2, D1, D2):
    identity_A = G2(real_A)
    identity_B = G1(real_B)
    
    g1_adv_loss = calc_mse_loss(D2(fake_B), 1.0)
    g2_adv_loss = calc_mse_loss(D1(fake_A), 1.0)
    #print("Adv loss: ", g1_adv_loss, g2_adv_loss)
    
    g1_identity_loss = F.l1_loss(identity_B, real_B)
    g2_identity_loss = F.l1_loss(identity_A, real_A)
    #print("Identity loss: ", g1_identity_loss, g2_identity_loss)
    
    fwd_cycle_loss = F.l1_loss(cycle_BA, real_A)
    bwd_cycle_loss = F.l1_loss(cycle_AB, real_B)
    #print("Cycle losses: ", fwd_cycle_loss, bwd_cycle_loss)
    
    loss = g1_adv_loss + g2_adv_loss + 10 * (fwd_cycle_loss + bwd_cycle_loss) + 5 * (g1_identity_loss + g2_identity_loss)
             
    #print("Gen loss: ", loss)
    loss.backward()
    return loss
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import torchvision.transforms as transforms
import functools


#Nvidia implementation
class AdaptiveInstanceNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(AdaptiveInstanceNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        # weight and bias are dynamically assigned
        self.weight = None
        self.bias = None
        # just dummy buffers, not used
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        assert self.weight is not None and self.bias is not None, "Please assign weight and bias before calling AdaIN!"
        b, c = x.size(0), x.size(1)
        running_mean = self.running_mean.repeat(b)
        running_var = self.running_var.repeat(b)

        # Apply instance norm
        x_reshaped = x.contiguous().view(1, b * c, *x.size()[2:])

        out = F.batch_norm(
            x_reshaped, running_mean, running_var, self.weight, self.bias,
            True, self.momentum, self.eps)

        return out.view(b, c, *x.size()[2:])

    def __repr__(self):
        return self.__class__.__name__ + '(' + str(self.num_features) + ')'

    
class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        # print(x.size())
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x
    
class LIN(nn.Module):
    
    def __init__(self, num_features, eps=1e-5):
        super(LIN, self).__init__()
        
        self.num_features = num_features
        self.eps = eps
        self.rho = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.gamma = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.beta = nn.Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)
    
    def forward(self, inputs):
        #inputs_in_mean, inputs_in_var = torch.mean(inputs, dim=[2, 3], keep_dim=True), torch.var(inputs, dim=[2, 3], keep_dim=True)
        inputs_in_mean = torch.mean(inputs.view(inputs.size(0), inputs.size(1), -1), 2).unsqueeze(2).unsqueeze(3)
        inputs_in_var = torch.var(inputs.view(inputs.size(0), inputs.size(1), -1), 2).unsqueeze(2).unsqueeze(3)
        inputs_in = (inputs - inputs_in_mean) / torch.sqrt((inputs_in_var + self.eps))
        #inputs_ln_mean, inputs_ln_var = torch.mean(inputs, dim[1, 2, 3], keep_dim=True), torch.var(inputs, dim=[1, 2, 3], keep_dim=True)
        inputs_ln_mean = torch.mean(inputs.view(inputs.size(0), -1), 1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        inputs_ln_var = torch.var(inputs.view(inputs.size(0), -1), 1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        inputs_ln = (inputs - inputs_ln_mean) / torch.sqrt((inputs_ln_var + self.eps))
        
        out = self.rho.expand(inputs.shape[0], -1, -1, -1) * inputs_in + \
                ((1 - self.rho).expand(inputs.shape[0], -1, -1, -1) * inputs_ln)
        
        return self.gamma.expand(inputs.shape[0], -1, -1, -1) * out + self.beta.expand(inputs.shape[0], -1, -1, -1)


class AdaLIN(nn.Module):
    
    def __init__(self, num_features, eps=1e-5):
        super(AdaLIN, self).__init__()
        
        self.num_features = num_features
        self.eps = eps
        self.rho = nn.Parameter(torch.empty(1, num_features, 1, 1))
        self.rho.data.fill_(0.9)
    
    def forward(self, inputs, gamma, beta):
        #inputs_in_mean, inputs_in_var = torch.mean(inputs, dim=[2, 3], keep_dim=True), torch.var(inputs, dim=[2, 3], keep_dim=True)
        inputs_in_mean = torch.mean(inputs.view(inputs.size(0), inputs.size(1), -1), 2).unsqueeze(2).unsqueeze(3)
        inputs_in_var = torch.var(inputs.view(inputs.size(0), inputs.size(1), -1), 2).unsqueeze(2).unsqueeze(3)
        inputs_in = (inputs - inputs_in_mean) / torch.sqrt((inputs_in_var + self.eps))
        #inputs_ln_mean, inputs_ln_var = torch.mean(inputs, dim[1, 2, 3], keep_dim=True), torch.var(inputs, dim=[1, 2, 3], keep_dim=True)
        inputs_ln_mean = torch.mean(inputs.view(inputs.size(0), -1), 1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        inputs_ln_var = torch.var(inputs.view(inputs.size(0), -1), 1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        inputs_ln = (inputs - inputs_ln_mean) / torch.sqrt((inputs_ln_var + self.eps))
        
        out = self.rho.expand(inputs.shape[0], -1, -1, -1) * inputs_in + \
              ((1 - self.rho).expand(inputs.shape[0], -1, -1, -1) * inputs_ln)
        
        return gamma.unsqueeze(2).unsqueeze(3) * out + beta.unsqueeze(2).unsqueeze(3)


class RhoClipper(object):

    def __init__(self, min, max):
        self.clip_min = min
        self.clip_max = max
        assert min < max

    def __call__(self, module):

        if hasattr(module, 'rho'):
            w = module.rho.data
            w = w.clamp(self.clip_min, self.clip_max)
            module.rho.data = w

class ImageBuffer():
    
    def __init__(self, size):
        self.buffer = []
        self.size = size

    def update(self, img):
        if len(self.buffer) < self.size:
            self.buffer.append(img)
            return img
        else:
            if random.random() < 0.5:
                idx = random.randint(0, self.size - 1)
                fake_img = self.buffer[idx]
                self.buffer[idx] = img
                return fake_img
            else:
                return img

class Identity(nn.Module):
    
    
    def __init__(self, channels):
        super(Identity, self).__init__()
        
    
    def forward(self, inputs):
        return inputs
    
    
def get_norm_module(name):
    if name == "batch":
        return nn.BatchNorm2d
    elif name == "instance":
        return nn.InstanceNorm2d
    elif name == "adain":
        return AdaptiveInstanceNorm2d
    elif name == "ln":
        return LayerNorm
    elif name == "lin":
        return LIN
    else:
        return Identity
    
def get_activation(name):
    if name == "relu":
        return F.relu
    elif name == "tanh":
        return F.tanh
    elif name == "lrelu":
        return functools.partial(F.leaky_relu, negative_slope=0.2)
    else:
        return None
    
def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

def init_weights_normal(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.Linear):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
    elif isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.normal_(m.bias, 0.0)


def calc_mse_loss(inputs, value=0):
    target = torch.Tensor((inputs.shape)).fill_(value).cuda()
    return F.mse_loss(inputs, target)

def tensor_to_image(tensor):
    std = torch.Tensor([0.5, 0.5, 0.5])
    mean = torch.Tensor([0.5, 0.5, 0.5])
    transf = transforms.Compose([transforms.Normalize(mean=(-mean/std).tolist(), std=(1.0/std).tolist()),
                                 transforms.ToPILImage()])
    return transf(tensor)

def save_cyclegan_model(G1, G2, D1, D2, path, num_epoch, postfix=''):
    torch.save(G1.state_dict(), path + 'g1_' + str(num_epoch) + postfix + '.pth')
    torch.save(G2.state_dict(), path + 'g2_' + str(num_epoch) + postfix + '.pth')
    torch.save(D1.state_dict(), path + 'd1_' + str(num_epoch) + postfix + '.pth')
    torch.save(D2.state_dict(), path + 'd2_' + str(num_epoch) + postfix + '.pth')
        
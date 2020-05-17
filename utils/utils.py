import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
import functools
import numpy as np


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

class ImageStack():
    
    def __init__(self, size):
        self.stack = {"real":[],
                      "fake": []}
        self.size = size
        self.curr_size = 0
    
    def update(self, imgs):
        if self.curr_size < self.size:
            if random.random() > 0.5:
                self.stack["real"].append(imgs[0])
                self.stack["real"].append(imgs[2])
                self.stack["fake"].append(imgs[3])
                self.stack["fake"].append(imgs[1])
                self.curr_size += 1

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
    classname = m.__class__.__name__
    if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    if classname.find('Batch') == 0 and hassatr(m, 'weight'):
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        if hassatr(m, 'bias') and m.bias is not None:
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


def load_vgg_feature_extractor():
    model = torchvision.models.vgg16(pretrained=True)
    vgg_feature_extractor = model.features[:23]
    vgg_feature_extractor.eval()
    
    for param in vgg_feature_extractor.parameters():
        param.requires_grad = False

    return vgg_feature_extractor


def preprocess_vgg(image):
    tensortype = type(image.data)
    (r, g, b) = torch.chunk(image, 3, dim = 1)
    image = torch.cat((b, g, r), dim = 1) # convert RGB to BGR
    image = (image + 1) * 255 * 0.5 # [-1, 1] -> [0, 255]
    mean = tensortype(image.data.size()).cuda()
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    image = image - mean # subtract mean
    return image


def calc_IN_feature_distance(feature_extractor, inputs):
    inputs_1 = preprocess_vgg(inputs[0])
    inputs_2 = preprocess_vgg(inputs[1])

    features_1 = feature_extractor(inputs_1)
    features_2 = feature_extractor(inputs_2)

    instance_norm = torch.nn.InstanceNorm2d(features_1.shape[2], affine=False)

    return torch.mean((instance_norm(features_1) - instance_norm(features_2))**2)

def from_tensor_to_image(tensor):
    tensor = tensor.detach().cpu().squeeze(0).numpy()
    tensor = tensor.transpose(1, 2, 0)
    return ((tensor + 1) * 255 * 0.5).astype(np.uint8)
        
import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import torchvision.transforms as transforms


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


def get_norm_module(name):
    if name == "batch":
        return nn.BatchNorm2d
    elif name == "instance":
        return nn.InstanceNorm2d
    else:
        return None
    
    
def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


def calc_mse_loss(inputs, value=0):
    target = torch.Tensor((inputs.shape)).fill_(value).cuda()
    return F.mse_loss(inputs, target)

def tensor_to_image(tensor):
    std = torch.Tensor([0.5, 0.5, 0.5])
    mean = torch.Tensor([0.5, 0.5, 0.5])
    transf = transforms.Compose([transforms.Normalize(mean=(-mean/std).tolist(), std=(1.0/std).tolist()),
                                 transforms.ToPILImage()])
    return transf(tensor)

def save_cyclegan_model(G1, G2, D1, D2, path, num_epoch):
    torch.save(G1.state_dict(), path + 'g1_' + str(num_epoch) + '.pth')
    torch.save(G2.state_dict(), path + 'g2_' + str(num_epoch) + '.pth')
    torch.save(D1.state_dict(), path + 'd1_' + str(num_epoch) + '.pth')
    torch.save(D2.state_dict(), path + 'd2_' + str(num_epoch) + '.pth')
        
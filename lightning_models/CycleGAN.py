import sys
sys.path.append('..')
import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from datasets.UnalignedDataset import UnalignedDataset
from torch.utils.data.dataloader import DataLoader
from utils.utils import calc_mse_loss, ImageStack
from models.CycleGAN import *
from argparse import ArgumentParser
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from collections import OrderedDict
import torchvision


class CycleGAN(LightningModule):
    
    def __init__(self, hparams):
        super().__init__()
        
        self.hparams = hparams
        self.model = CycleGAN_pytorch(hparams.in_channels, hparams.n_blocks, hparams.norm_type_gen,
                                      hparams.norm_type_discr)
        self.last_imgs = None
        self.val_stack = ImageStack(8)
    
    def forward(self, inputs):
        return self.G1(inputs)
        
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        #parser.add_argument('--gpus', type=int, default=1)
        parser.add_argument('--lr', type=float, default=0.0002)
        parser.add_argument('--beta_1', type=float, default=0.5)
        parser.add_argument('--beta_2', type=float, default=0.99)
        parser.add_argument('--in_channels', type=int, default=3)
        parser.add_argument('--n_blocks', type=int, default=9)
        parser.add_argument('--norm_type_gen', type=str, default='instance')
        parser.add_argument('--norm_type_discr', type=str, default='instance')
        parser.add_argument('--resize', type=int, default=268)
        parser.add_argument('--crop', type=int, default=256)
        parser.add_argument('--limit', type=int, default=50)
        parser.add_argument('--batch_size', type=int, default=1)
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--shuffle', type=bool, default=True)
        parser.add_argument('--folder_names', nargs="+")
        parser.add_argument('--val_split', type=float, default=0.1)
        parser.add_argument('--root', type=str, default='/content/drive/My Drive/')

        return parser
        
    def training_step(self, batch, batch_nb, optimizer_idx):
        self.last_imgs = batch

        real_A = batch["A"]
        real_B = batch["B"]
        
        if optimizer_idx == 0:

            fake_B = self.model.G1(real_A)
            cycle_BA = self.model.G2(fake_B)
            fake_A = self.model.G2(real_B)
            cycle_AB = self.model.G1(fake_A)

            loss = self.model.backward_Gs(fake_B, cycle_BA, fake_A, cycle_AB, real_A, real_B)

            tqdm_dict = {'g_loss': loss}
            output = OrderedDict({
                'loss': loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            
            return output
       
        if optimizer_idx == 1:

            fake_B = self.model.G1(real_A)
            fake_A = self.model.G2(real_B)
            
            loss = self.model.backward_Ds(real_A, real_B, fake_A, fake_B)
            
            tqdm_dict = {'d_loss': loss}
            output = OrderedDict({
                'loss': loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
           
            return output
      
    def configure_optimizers(self):
        lr = self.hparams.lr
        beta_1 = self.hparams.beta_1
        beta_2 = self.hparams.beta_2
        
        
        optimizer_g = optim.Adam(list(self.model.G1.parameters()) + list(self.model.G2.parameters()),
                                 lr=lr, betas=(beta_1, beta_2))
        optimizer_d = optim.Adam(list(self.model.D1.parameters()) + list(self.model.D2.parameters()),
                                 lr=lr, betas=(beta_1, beta_2))
        
        return [optimizer_g, optimizer_d], []
    

    def prepare_data(self):
        transform = transforms.Compose([transforms.Resize((self.hparams.resize, self.hparams.resize), Image.BICUBIC),
                        transforms.RandomCrop(self.hparams.crop),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        dataset_train = UnalignedDataset(self.hparams.root, self.hparams.folder_names, self.hparams.limit, transform)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset_train,
                                                                   [round(len(dataset_train)*(1 - self.hparams.val_split)),
                                                                   round(len(dataset_train)* self.hparams.val_split)])

        
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=self.hparams.shuffle,
                          num_workers=self.hparams.num_workers)
    
    def on_epoch_end(self):
        real_A = self.last_imgs["A"]
        real_B = self.last_imgs["B"]

        fake_A = self.model.G2(real_B.cuda())
        fake_B = self.model.G1(real_A.cuda())
 
        print(real_A.shape, real_B.shape, fake_A.shape, fake_B.shape)
        grid = torchvision.utils.make_grid(torch.cat([real_A, real_B, fake_B, fake_A], dim=0), 
                                            nrow=2, normalize=True, range=(-1.0, 1.0), scale_each=True)
        self.logger.experiment.add_image(f'Real Domains and Fake', grid, self.current_epoch)
    
    
    def validation_step(self, batch, batch_idx):
        real_A = batch["A"]
        real_B = batch["B"]
        
        fake_A = self.model.G2(real_B.cuda())
        fake_B = self.model.G1(real_A.cuda())
        cycle_BA = self.model.G2(fake_B)
        cycle_AB = self.model.G1(fake_A)

        val_loss = self.model.backward_Gs(fake_B, cycle_BA, fake_A, cycle_AB, real_A, real_B)

        tqdm_dict = {'g_val_loss': val_loss}

        self.val_stack.update([real_A, fake_A, real_B, fake_B])

        return tqdm_dict
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers)        


    def validation_epoch_end(self, outputs):
       real = self.val_stack.stack["real"]
       fake = self.val_stack.stack["fake"]
       #print(real, fake)
       grid = torchvision.utils.make_grid(torch.cat(real[:8] + fake[:8] + real[8:16] + fake[8:16],dim=0),
                                          nrow=8, normalize=True, range=(-1.0, 1.0), scale_each=True)

       self.logger.experiment.add_image(f'Real Domains and Fake val', grid, self.current_epoch)
       self.val_stack = ImageStack(8)

       avg_loss = torch.stack([x['g_val_loss'] for x in outputs]).mean()
       tqdm_dict = {'g_val_loss': avg_loss}
       return {'val_loss': avg_loss, 'log': tqdm_dict}
    
        
        
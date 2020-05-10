import sys
sys.path.append('..')
import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from datasets.UnalignedDataset import UnalignedDataset
from torch.utils.data.dataloader import DataLoader
from utils.utils import calc_mse_loss
from models.CycleGAN import PatchGan, ResnetGenerator
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
        self.G1 = ResnetGenerator(hparams.in_channels, hparams.n_blocks, hparams.norm_type_gen)
        self.G2 = ResnetGenerator(hparams.in_channels, hparams.n_blocks, hparams.norm_type_gen)
        self.D1 = PatchGan(hparams.in_channels, hparams.norm_type_discr)
        self.D2 = PatchGan(hparams.in_channels, hparams.norm_type_discr)
        self.last_imgs = None
        #self.model = CycleGAN_pytorch(hparams)
    
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
        parser.add_argument('--root', type=str, default='/content/drive/My Drive/')
        return parser
        
    def training_step(self, batch, batch_nb, optimizer_idx):
        self.last_imgs = batch

        real_A = batch["A"]
        real_B = batch["B"]
        
        
        fake_B = self.G1(real_A)
        cycle_BA = self.G2(fake_B)
        fake_A = self.G2(real_B)
        cycle_AB = self.G1(fake_A)
        #fake_B, cycle_BA, fake_A, cycle_AB = calc_Gs_outputs(self.model.G1, self.model.G2, domain_A, domain_B)
        
        if optimizer_idx == 0:
            
            identity_A = self.G2(real_A)
            identity_B = self.G1(real_B)

            g1_adv_loss = calc_mse_loss(self.D2(fake_B), 1.0)
            g2_adv_loss = calc_mse_loss(self.D1(fake_A), 1.0)
            #print("Adv loss: ", g1_adv_loss, g2_adv_loss)

            g1_identity_loss = F.l1_loss(identity_B, real_B)
            g2_identity_loss = F.l1_loss(identity_A, real_A)
            #print("Identity loss: ", g1_identity_loss, g2_identity_loss)

            fwd_cycle_loss = F.l1_loss(cycle_BA, real_A)
            bwd_cycle_loss = F.l1_loss(cycle_AB, real_B)
            #print("Cycle losses: ", fwd_cycle_loss, bwd_cycle_loss)

            loss = g1_adv_loss + g2_adv_loss + 10 * (fwd_cycle_loss + bwd_cycle_loss) + 5 * (g1_identity_loss + g2_identity_loss)
            tqdm_dict = {'g_loss': loss}
            output = OrderedDict({
                'loss': loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            
            return output
       
        if optimizer_idx == 1:
            real_output = self.D1(real_A)
            #print(real_output.shape)
            #print(real_output)
            d1_real_loss = calc_mse_loss(real_output, 1.0)
            #F.mse_loss(real_output, torch.ones(real_output.shape).cuda())

            fake_output = self.D1(fake_A.detach())
            d1_fake_loss = F.mse_loss(fake_output, torch.zeros(fake_output.shape).cuda())
            
            real_output = self.D2(real_B)
            
            d2_real_loss = calc_mse_loss(real_output, 1.0)
            
            fake_output = self.D2(fake_B.detach())
            d2_fake_loss = F.mse_loss(fake_output, torch.zeros(fake_output.shape).cuda())

            loss = (d1_fake_loss + d1_real_loss + d2_fake_loss + d2_real_loss) * 0.5
            
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
        
        
        optimizer_g = optim.Adam(list(self.G1.parameters()) + list(self.G2.parameters()),
                                 lr=lr, betas=(beta_1, beta_2))
        optimizer_d = optim.Adam(list(self.D1.parameters()) + list(self.D2.parameters()),
                                 lr=lr, betas=(beta_1, beta_2))
        
        return [optimizer_g, optimizer_d], []
    
    
    def train_dataloader(self):
        transform = transforms.Compose([transforms.Resize((self.hparams.resize, self.hparams.resize), Image.BICUBIC),
                                transforms.RandomCrop(self.hparams.crop),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        dataset_train = UnalignedDataset(self.hparams.root, self.hparams.folder_names, self.hparams.limit, transform)
        print(len(dataset_train))
        return DataLoader(dataset_train, batch_size=self.hparams.batch_size, shuffle=self.hparams.shuffle,
                          num_workers=self.hparams.num_workers)
    
    def on_epoch_end(self):
        real_A = self.last_imgs["A"]
        real_B = self.last_imgs["B"]

        fake_A = self.G2(real_B.cuda())
        fake_B = self.G1(real_A.cuda())
 
        print(real_A.shape, real_B.shape, fake_A.shape, fake_B.shape)
        grid = torchvision.utils.make_grid(torch.cat([real_A, real_B, fake_B, fake_A], dim=0), 
                                            nrow=2, normalize=True, range=(-1.0, 1.0), scale_each=True)
        self.logger.experiment.add_image(f'Real Domains and Fake', grid, self.current_epoch)
    
    
    def validation_step(self):
        pass
    
    def val_dataloader(self):
        pass

    def validation_epoch_end(self):
        pass
    
        
        
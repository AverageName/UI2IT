import sys
sys.path.append('..')
import torch
from torch.nn import functional as F
from torch import nn
from pytorch_lightning.core.lightning import LightningModule
from datasets.UnalignedDataset import UnalignedDataset
from torch.utils.data.dataloader import DataLoader
from utils.utils import calc_mse_loss
from models.UGATIT import UGATIT_pytorch
from argparse import ArgumentParser
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from collections import OrderedDict
import torchvision



class UGATIT(LightningModule):


    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        print(hparams)
        self.last_imgs = None
        self.model = UGATIT_pytorch(hparams.in_channels, hparams.crop, hparams.num_enc_blocks,
                            hparams.num_enc_res_blocks, hparams.num_dec_upsample_blocks,
                            hparams.num_dec_res_blocks, hparams.norm_type, hparams.pad_type,
                            hparams.local_discr_num_downsample, hparams.global_discr_num_downsample)
    

    def forward(self, domain_A, domain_B):
        return self.model(domain_A, domain_B)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=0.0001)
        parser.add_argument('--beta_1', type=float, default=0.5)
        parser.add_argument('--beta_2', type=float, default=0.99)
        parser.add_argument('--in_channels', type=int, default=3)
        parser.add_argument('--num_enc_blocks', type=int, default=3)
        parser.add_argument('--num_enc_res_blocks', type=int, default=4)
        parser.add_argument('--num_dec_res_blocks', type=int, default=4)
        parser.add_argument('--num_dec_upsample_blocks', type=int, default=2)
        parser.add_argument('--local_discr_num_downsample', type=int, default=3)
        parser.add_argument('--global_discr_num_downsample', type=int, default=5)
        parser.add_argument('--norm_type', type=str, default='instance')
        parser.add_argument('--pad_type', type=str, default='reflection')
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
        domain_A = batch["A"]
        domain_B = batch["B"]

        if optimizer_idx == 0:
            loss = self.model.backward_Gs(domain_A, domain_B)
            self.model.G_AB.apply(self.model.rho_clipper)
            self.model.G_BA.apply(self.model.rho_clipper)

            tqdm_dict = {'g_loss': loss}
            output = OrderedDict({
                'loss': loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            
            return output
        
        if optimizer_idx == 1:
            loss = self.model.backward_Ds(domain_A, domain_B)
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

        optimizer_g = optim.Adam(list(self.model.G_AB.parameters()) + list(self.model.G_BA.parameters()),
                                 lr=lr, betas=(beta_1, beta_2))
        optimizer_d = optim.Adam(list(self.model.D_AL.parameters()) + list(self.model.D_AG.parameters()) + \
                                 list(self.model.D_BL.parameters()) + list(self.model.D_BG.parameters()),
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

        fake_A, _ = self.model.G_BA(real_B.cuda())
        fake_B, _ = self.model.G_AB(real_A.cuda())

        print(real_A.shape, real_B.shape, fake_A.shape, fake_B.shape)
        grid = torchvision.utils.make_grid(torch.cat([real_A, real_B, fake_B, fake_A], dim=0), 
                                            nrow=2, normalize=True, range=(-1.0, 1.0), scale_each=True)
        self.logger.experiment.add_image(f'Real Domains and Fake', grid, self.current_epoch)

    
    #def validation_step(self):
    #    pass
    
    #def val_dataloader(self):
    #    pass

    #def validation_epoch_end(self, outputs):
    #    pass
    




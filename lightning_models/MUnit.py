import sys
sys.path.append('..')
from pytorch_lightning.core.lightning import LightningModule
from models.MUnit import *
from argparse import ArgumentParser
from collections import OrderedDict
from utils.utils import ImageStack

class MUnit(LightningModule):


    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

        self.model = MUnit_pytorch(hparams.in_channels, hparams.mlp_hidden_dim, hparams.mlp_num_blocks, hparams.d_num_scales, 
                                   hparams.enc_style_dims, hparams.enc_cont_num_blocks, hparams.norm_type_cont, hparams.pad_type_cont, 
                                   hparams.norm_type_style, hparams.pad_type_style, hparams.norm_type_decoder, hparams.pad_type_decoder,
                                   hparams.norm_type_mlp, hparams.enc_cont_dim, hparams.use_perceptual_loss)
        
        self.last_imgs = None
        self.val_stack = ImageStack(8)
        self.every_amount_epochs = None
        print("INITIAL LR: ", hparams.lr)
        self.curr_lr = hparams.lr
    
    def forward(self, domain_A, domain_B):
        return self.model(domain_A, domain_B)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=0.0002)
        parser.add_argument('--beta_1', type=float, default=0.5)
        parser.add_argument('--beta_2', type=float, default=0.99)
        parser.add_argument('--in_channels', type=int, default=3)
        parser.add_argument('--mlp_hidden_dim', type=int, default=256)
        parser.add_argument('--mlp_num_blocks', type=int, default=3)
        parser.add_argument('--d_num_scales', type=int, default=3)
        parser.add_argument('--enc_style_dims', type=int, default=8)
        parser.add_argument('--enc_cont_num_blocks', type=int, default=4)
        parser.add_argument('--norm_type_cont', type=str, default='instance')
        parser.add_argument('--pad_type_cont', type=str, default='reflection')
        parser.add_argument('--norm_type_style', type=str, default='none')
        parser.add_argument('--pad_type_style', type=str, default='reflection')
        parser.add_argument('--norm_type_decoder', type=str, default='adain')
        parser.add_argument('--pad_type_decoder', type=str, default='reflection')
        parser.add_argument('--norm_type_mlp', type=str, default='none')
        parser.add_argument('--enc_cont_dim', type=int, default=256)
        parser.add_argument('--use_perceptual_loss', type=bool, default=False)
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

        domain_A = batch["A"]
        domain_B = batch["B"]

        if optimizer_idx == 0:
            loss = self.model.backward_Gs(domain_A, domain_B)
            tqdm_dict = {'g_loss': loss}
            output = OrderedDict({
                'loss': loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            
            return output

        elif optimizer_idx == 1:

            fake_A, fake_B = self.model.forward(domain_A, domain_B)
            loss = self.model.backward_Ds(domain_A, fake_A, domain_B, fake_B)

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

        self.every_amount_epochs = 100000 // len(self.train_dataset)
        lr_lambda = lambda epoch: 1 / 2**(epoch // self.every_amount_epochs)
        scheduler_g = optim.lr_scheduler.LambdaLR(optimizer_g, lr_lambda=lr_lambda)
        scheduler_d = optim.lr_scheduler.LambdaLR(optimizer_d, lr_lambda=lr_lambda)

        return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]


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

        fake_A, fake_B = self.model(real_A, real_B)
 
        print(real_A.shape, real_B.shape, fake_A.shape, fake_B.shape)
        grid = torchvision.utils.make_grid(torch.cat([real_A, real_B, fake_B, fake_A], dim=0), 
                                            nrow=2, normalize=True, range=(-1.0, 1.0), scale_each=True)
        self.logger.experiment.add_image(f'Real Domains and Fake', grid, self.current_epoch)


    def validation_step(self, batch, batch_idx):
        real_A = batch["A"]
        real_B = batch["B"]
        
        fake_A, fake_B = self.model(real_A, real_B)

        val_loss = self.model.backward_Gs(real_A, real_B)

        tqdm_dict = {'g_val_loss': val_loss}

        self.val_stack.update([real_A, fake_A, real_B, fake_B])

        return tqdm_dict
    

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers)        


    def validation_epoch_end(self, outputs):
       real = self.val_stack.stack["real"]
       fake = self.val_stack.stack["fake"]


       grid_data = []
       for i in range(len(real) // 2):
           grid_data += real[2 * i: 2 * (i + 1)]
           grid_data += fake[2 * i: 2 * (i + 1)]

       grid = torchvision.utils.make_grid(torch.cat(grid_data, dim=0),
                                          nrow=2, normalize=True, range=(-1.0, 1.0), scale_each=True)

       self.logger.experiment.add_image(f'Real Domains and Fake val', grid, self.current_epoch)
       self.val_stack = ImageStack(8)

       avg_loss = torch.stack([x['g_val_loss'] for x in outputs]).mean()
       tqdm_dict = {'g_val_loss': avg_loss}
       return {'val_loss': avg_loss, 'log': tqdm_dict}
    

    @staticmethod
    def predict_dataloader(args):

        transform = transforms.Compose([transforms.Resize((args.resize, args.resize), Image.BICUBIC),
                        transforms.RandomCrop(args.crop),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        dataset_predict = UnalignedDataset(args.root, args.folder_names, args.limit, transform)
        dataloader_predict = DataLoader(dataset_predict, batch_size=1, num_workers=args.num_workers)

        return dataloader_predict
    
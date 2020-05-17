import os
from argparse import ArgumentParser
from lightning_models.CycleGAN import CycleGAN
from lightning_models.UGATIT import UGATIT
from lightning_models.MUnit import MUnit
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.utils import init_weights_normal
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main(args):

    if args.model_name == 'cyclegan':
        model = CycleGAN(hparams=args)
    elif args.model_name == 'ugatit':
        model = UGATIT(hparams=args)
    elif args.model_name == 'munit':
        model = MUnit(hparams=args)
    
    checkpoint_callback = ModelCheckpoint(
                        filepath=args.save_checkpoint_path,
                        save_top_k=1,
                        verbose=True,
                        monitor='g_loss',
                        mode='min',
                        prefix=''
                        )

    print(args.load_checkpoint_path)
    if args.load_checkpoint_path is None:
        model.apply(init_weights_normal)
        trainer = Trainer(gpus=args.gpus, max_epochs=args.max_epochs, 
                          checkpoint_callback=checkpoint_callback)
    else:
        if args.resume_training:
            trainer = Trainer(gpus=args.gpus, max_epochs=args.max_epochs, 
                      resume_from_checkpoint=args.load_checkpoint_path,
                      checkpoint_callback=checkpoint_callback)
        else:
            model.load_from_checkpoint(args.load_checkpoint_path)
            trainer = Trainer(gpus=args.gpus, max_epochs=args.max_epochs, 
                          checkpoint_callback=checkpoint_callback)

    trainer.fit(model)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)

    parser.add_argument("--model_name", type=str, default='cyclegan')
    parser.add_argument("--save_checkpoint_path", type=str, default='./checkpoints/')
    parser.add_argument("--load_checkpoint_path", type=str)
    parser.add_argument("--resume_training", type=bool, default=False)

    temp_args, _ = parser.parse_known_args()
    if temp_args.model_name == 'cyclegan':
        parser = CycleGAN.add_model_specific_args(parser)
    elif temp_args.model_name == 'ugatit':
        parser = UGATIT.add_model_specific_args(parser)
    elif temp_args.model_name == 'munit':
        parser = MUnit.add_model_specific_args(parser)

    args = parser.parse_args()

    main(args)
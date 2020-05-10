from argparse import ArgumentParser
from lightning_models.CycleGAN import CycleGAN
from lightning_models.UGATIT import UGATIT
from pytorch_lightning import Trainer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main(args):

    if args.model_name == 'cyclegan':
        model = CycleGAN(hparams=args)
    elif args.model_name == 'ugatit':
        model = UGATIT(hparams=args)
    
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)

    parser.add_argument("--model_name", type=str, default='cyclegan')

    temp_args, _ = parser.parse_known_args()
    if temp_args.model_name == 'cyclegan':
        parser = CycleGAN.add_model_specific_args(parser)
    elif temp_args.model_name == 'ugatit':
        parser = UGATIT.add_model_specific_args(parser)

    args = parser.parse_args()

    main(args)
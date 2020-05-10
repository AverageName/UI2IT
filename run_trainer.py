from argparse import ArgumentParser
from lightning_models.CycleGAN import CycleGAN
from pytorch_lightning import Trainer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


parser = ArgumentParser()
parser = CycleGAN.add_model_specific_args(parser)
parser = Trainer.add_argparse_args(parser)

hparams = parser.parse_args()
#print(hparams.folder_names)
model = CycleGAN(hparams)
trainer = Trainer.from_argparse_args(hparams)
trainer.fit(model)
import pytest
import os
print(os.getcwd())
from argparse import ArgumentParser, Namespace
from lightning_models.CycleGAN import CycleGAN
from lightning_models.UGATIT import UGATIT
from lightning_models.MUnit import MUnit
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.utils import init_weights_normal
import yaml

class Test_training:

    @pytest.mark.timeout(180)
    def test_cycle(self):
        with open('./test_configs/test_cyclegan_train.yaml', "r") as f:
            dict_ = yaml.safe_load(f)
        args = Namespace(**dict_)
        model = CycleGAN(hparams=args)

        checkpoint_callback = ModelCheckpoint(
                        filepath='./checkpoints/',
                        save_top_k=1,
                        verbose=True,
                        monitor='g_loss',
                        mode='min',
                        prefix=''
                        )

        model.apply(init_weights_normal)
        trainer = Trainer(gpus=args.gpus, max_epochs=args.max_epochs, 
                        checkpoint_callback=checkpoint_callback)

        trainer.fit(model)
    
    @pytest.mark.timeout(180)
    def test_munit(self):
        with open('./test_configs/test_munit_train.yaml', "r") as f:
            dict_ = yaml.safe_load(f)
        args = Namespace(**dict_)

        model = MUnit(hparams=args)

        checkpoint_callback = ModelCheckpoint(
                        filepath='./checkpoints/',
                        save_top_k=1,
                        verbose=True,
                        monitor='g_loss',
                        mode='min',
                        prefix=''
                        )

        model.apply(init_weights_normal)
        trainer = Trainer(gpus=args.gpus, max_epochs=args.max_epochs, 
                        checkpoint_callback=checkpoint_callback)

        trainer.fit(model)


    @pytest.mark.timeout(180)
    def test_ugatit(self):
        with open('./test_configs/test_ugatit_train.yaml', "r") as f:
            dict_ = yaml.safe_load(f)
        args = Namespace(**dict_)

        model = UGATIT(hparams=args)

        checkpoint_callback = ModelCheckpoint(
                        filepath='./checkpoints/',
                        save_top_k=1,
                        verbose=True,
                        monitor='g_loss',
                        mode='min',
                        prefix=''
                        )

        model.apply(init_weights_normal)
        trainer = Trainer(gpus=args.gpus, max_epochs=args.max_epochs, 
                        checkpoint_callback=checkpoint_callback)

        trainer.fit(model)
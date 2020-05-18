import os
from argparse import ArgumentParser, Namespace
from lightning_models.CycleGAN import CycleGAN
from lightning_models.UGATIT import UGATIT
from lightning_models.MUnit import MUnit
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from utils.utils import init_weights_normal, from_tensor_to_image
from datasets.UnalignedDataset import UnalignedDataset
from PIL import Image
import yaml
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(args):

    if args.model_name == 'cyclegan':
        model = CycleGAN.load_from_checkpoint(args.load_checkpoint_path)
    elif args.model_name == 'ugatit':
        model = UGATIT.load_from_checkpoint(args.load_checkpoint_path)
    elif args.model_name == 'munit':
        model = MUnit.load_from_checkpoint(args.load_checkpoint_path)
    
    dataloader = model.predict_dataloader(args)

    dirs = [os.path.join(args.root, args.model_name + "_gen_A"), os.path.join(args.root, args.model_name + "_gen_B")]
    for dir in dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)

    for idx, data in enumerate(dataloader):
        fake_A, fake_B = model(data["A"], data["B"])
        #print(from_tensor_to_image(fake_A))
        image_A = Image.fromarray(from_tensor_to_image(fake_A))
        image_B = Image.fromarray(from_tensor_to_image(fake_B))
        image_A.save(os.path.join(dirs[0], str(idx) + ".jpg"))
        image_B.save(os.path.join(dirs[1], str(idx) + ".jpg"))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)

    parser.add_argument("--model_name", type=str, default='cyclegan')
    parser.add_argument("--root", type=str)
    parser.add_argument("--folder_names", nargs="+")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--load_checkpoint_path", type=str)
    parser.add_argument("--crop", type=int, default=256)
    parser.add_argument("--resize", type=int, default=286)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--yaml_path", type=str)

    temp_args, _ = parser.parse_known_args()

    # if temp_args.model_name == 'cyclegan':
    #     parser = CycleGAN.add_model_specific_args(parser)
    # elif temp_args.model_name == 'ugatit':
    #     parser = UGATIT.add_model_specific_args(parser)
    # elif temp_args.model_name == 'munit':
    #     parser = MUnit.add_model_specific_args(parser)

    if temp_args.yaml_path is not None:
        with open(temp_args.yaml_path, "r") as f:
            dict_ = yaml.safe_load(f)
        args = Namespace(**dict_)
    else:
        args = parser.parse_args()

    main(args)
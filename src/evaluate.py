"""Evaluate module"""

import argparse
import numpy as np

import torch
import yaml
from yaml import CLoader as Loader
import segmentation_models_pytorch as smp
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import SegmentationDataset, get_transforms_segmentation
from src.models import EnhancementModel
from utils import ConvNormActBlock, seed_everything


def parse_args():
    """Parse command line arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="path to config")
    parser.add_argument(
        "--weights_path_enhancement",
        type=str,
        required=True,
        help="path to enhancement model weights",
    )
    parser.add_argument(
        "--weights_path_segmentation",
        type=str,
        required=True,
        help="path to segmentation model weights",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config_path, encoding="utf-8") as f:
        config = yaml.load(f, Loader=Loader)

    seed_everything(seed=config["common_params"]["random_seed"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_dataset = SegmentationDataset(
        mode="test", transform=get_transforms_enhancement(), **config["dataset_params"]
    )
    test_dataloader = DataLoader(
        test_dataset,
        pin_memory=True,
        shuffle=False,
        **config["training_params"]["dataloader_params"]
    )

    enhancement_model = EnhancementModel(
        norm_layer=nn.BatchNorm2d,
        act_layer="lrelu",
        conv_block=ConvNormActBlock,
        **config["encoder_params"]
    )
    enhancement_model.to(device)
    enhancement_model.eval()

    enhancement_model.load_state_dict(torch.load(args.weights_path_enhancement))

    segmentation_model = smp.create_model(**config["segmentation_model_params"])
    segmentation_model.to(device)
    segmentation_model.eval()

    segmentation_model.load_state_dict(torch.load(args.weights_path_segmentation))

    masks = []
    for source, target in tqdm(test_dataloader):
        source, target = source.to(device), target.to(device)

        with torch.no_grad():
            enhanced_input = enhancement_model(source).detach()
            output = segmentation_model(enhanced_input)

        masks.append(output.cpu().numpy().squeeze())

    masks = np.stack(masks, axis=0)
    np.savez_compressed("masks.npz", masks=masks)


if __name__ == "__main__":
    main()

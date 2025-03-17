"""Main module to run experiments"""

import argparse
import logging
import os

import torch
import yaml
from yaml import CLoader as Loader
from piq import SSIMLoss
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from datasets import EnhancementDataset, get_transforms_enhancement
from enhancement_models import EnhancementModel
from train import train_enhancement_model
from utils import ConvNormActBlock, seed_everything

logging.basicConfig(
    filename="train_enhancement_model.log",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
)
logger = logging.getLogger()


def parse_args():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="path to config")
    return parser.parse_args()


def main():
    """Define variables and start train process"""

    args = parse_args()

    with open(args.config_path, encoding="utf-8") as f:
        config = yaml.load(f, Loader=Loader)

    os.makedirs(
        config["training_params"]["training_process"]["save_path"], exist_ok=True
    )

    seed_everything(seed=config["common_params"]["random_seed"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset = EnhancementDataset(
        mode="train", transform=get_transforms_enhancement(), **config["dataset_params"]
    )
    test_dataset = EnhancementDataset(
        mode="test", transform=get_transforms_enhancement(), **config["dataset_params"]
    )

    train_dataloader = DataLoader(
        train_dataset,
        pin_memory=True,
        shuffle=True,
        **config["training_params"]["dataloader_params"]
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

    logger.info(enhancement_model)
    logger.info(summary(enhancement_model, input_size=(3, 500, 500)))

    criterions = dict()
    criterions["L1"] = nn.L1Loss()
    criterions["SSIM"] = SSIMLoss(data_range=1.0)
    optimizer = torch.optim.AdamW(
        enhancement_model.parameters(),
        lr=float(config["training_params"]["optimizer_params"]["lr"]),
        betas=(
            float(config["training_params"]["optimizer_params"]["beta1"]),
            float(config["training_params"]["optimizer_params"]["beta2"]),
        ),
        eps=float(config["training_params"]["optimizer_params"]["eps"]),
        weight_decay=float(
            config["training_params"]["optimizer_params"]["weight_decay"]
        ),
        amsgrad=False,
    )

    writer = SummaryWriter(
        os.path.join(
            config["common_params"]["tensorboard_path"],
            os.path.basename(args.config_path)[:-5],
        )
    )

    train_enhancement_model(
        train_loader=train_dataloader,
        test_loader=test_dataloader,
        model=enhancement_model,
        criterions=criterions,
        optimizer=optimizer,
        device=device,
        writer=writer,
        **config["training_params"]["training_process"]
    )


if __name__ == "__main__":
    main()

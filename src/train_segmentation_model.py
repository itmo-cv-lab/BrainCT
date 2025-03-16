"""Train segmentation module"""

import argparse
import os

import torch
import yaml
from yaml import CLoader as Loader
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import segmentation_models_pytorch as smp

from datasets import SegmentationDataset, get_transforms_segmentation
from models import SegmentationModel
from train import train_segmentation_model
from utils import seed_everything


def parse_args():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True, help="path to config")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    with open(args.config_path, encoding="utf-8") as f:
        config = yaml.load(f, Loader=Loader)

    os.makedirs(
        config["training_params"]["training_process"]["save_path"], exist_ok=True
    )

    seed_everything(seed=config["common_params"]["random_seed"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset = SegmentationDataset(
        mode="train", transform=get_transforms_segmentation(), **config["dataset_params"]
    )
    test_dataset = SegmentationDataset(
        mode="test", transform=get_transforms_segmentation(), **config["dataset_params"]
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

    model = smp.create_model(**config["segmentation_model_params"])
    model.to(device)

    print(model)

    criterions = {}
    criterions["DiceLoss"] = smp.losses.DiceLoss(**config["segmentation_loss_params"])
    optimizer = torch.optim.AdamW(
        enhancement_model.parameters(),
        lr=float(config["training_params"]["optimizer_params"]["lr"]),
        betas=(float(config["training_params"]["optimizer_params"]["beta1"]),
               float(config["training_params"]["optimizer_params"]["beta2"])),
        eps=float(config["training_params"]["optimizer_params"]["eps"]),
        weight_decay=float(config["training_params"]["optimizer_params"]["weight_decay"]),
        amsgrad=False,
    )

    writer = SummaryWriter(
        os.path.join(
            config["common_params"]["tensorboard_path"],
            os.path.basename(args.config_path)[:-5],
        )
    )

    train_segmentation_model(
        train_loader=train_dataloader,
        test_loader=test_dataloader,
        model=model,
        criterions=criterions,
        optimizer=optimizer,
        device=device,
        writer=writer,
        **config["training_params"]["training_process"]
    )


if __name__ == "__main__":
    main()

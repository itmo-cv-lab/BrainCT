"""Training module"""

import copy
import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

logging.basicConfig(
    filename="train.log",
    level=logging.INFO,
    format="%(asctime)s %(message)s",
)
logger = logging.getLogger()


def train_enhancement_model(
    n_epochs,
    total_iters,
    train_loader,
    test_loader,
    model,
    criterions,
    optimizer,
    lr_decay_frequency,
    save_frequency,
    save_path,
    device,
    writer,
):
    """Train enhancement model function"""

    logger.info(f"Training started with epochs={n_epochs}")

    iters = 0
    for epoch in tqdm(range(n_epochs)):
        model.train()

        loss_dict = {}
        for k in criterions.keys():
            loss_dict[k] = []
        loss_dict["total_loss"] = []

        for source, target in tqdm(train_loader):
            source, target = source.to(device), target.to(device)
            output = model(source)

            loss = 0
            for k in criterions.keys():
                if k == "val":
                    continue
                cur_loss = criterions[k](output, target)
                loss += cur_loss
                loss_dict[k].append(cur_loss.item())

            loss_dict["total_loss"].append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mean_total_loss = np.mean(np.array(loss_dict["total_loss"]))
            logger.info(
                f"\nTrain, iter: {iters}, total_loss: {mean_total_loss}", end=", "
            )
            writer.add_scalar("total_loss", mean_total_loss, iters)

            if iters % lr_decay_frequency == 0:
                for g in optimizer.param_groups:
                    g["lr"] *= 0.95

            val_loss = validate_pipeline_enhancement(
                model, test_loader, criterions["val"], device
            )
            logger.info(f"Epoch {epoch+1}/{n_epochs}, Val Loss: {val_loss:.4f}")

            if iters % save_frequency == 0 or iters == total_iters:
                torch.save(model.state_dict(), save_path)
                logger.info("Model saved!")

            if iters == total_iters:
                break
            iters += 1

        if iters == total_iters:
            break

    logger.info("Training finished.")


def validate_pipeline_enhancement(model, val_loader, criterion, device):
    """Validation enhancement"""

    model.eval()

    total_loss = 0
    with torch.no_grad():
        for source, target in tqdm(val_loader):
            source, target = source.to(device), target.to(device)
            outputs = model(source)
            loss = criterion(outputs, target)
            total_loss += loss.item()

    return total_loss / len(val_loader.dataset)


def validate_pipeline_segmentation(model, val_loader, criterion):
    """Validation segmentation"""

    model.eval()

    total_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            images, masks = batch["image"].cuda(), batch["mask"].cuda()
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item() * images.size(0)

    return total_loss / len(val_loader.dataset)


def train_segmentation_model(
    segmentation_model, train_loader, val_loader, epochs=10, lr=1e-4, weight_decay=0.05
):
    """Train segmentation model function"""

    model = segmentation_model.to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    logger.info("Segmentation training started")
    criterion = (
        nn.CrossEntropyLoss()
        if segmentation_model.model.classes > 1
        else nn.BCEWithLogitsLoss()
    )

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        for batch in train_loader:
            images, masks = batch["image"].cuda(), batch["mask"].cuda()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

        val_loss = validate_pipeline_segmentation(model, val_loader, criterion)
        logger.info(
            f"[Segmentation] Epoch {epoch+1}/{epochs}, Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            logger.info(
                f"[Segmentation] New best model saved at epoch {epoch+1} with loss {val_loss:.4f}"
            )

    logger.info(
        f"Segmentation training finished. Best validation loss: {best_loss:.4f}"
    )
    model.load_state_dict(best_model_wts)

    return model

"""Training module"""

import numpy as np
import torch
from tqdm import tqdm


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

    iters = 0

    for _ in tqdm(range(n_epochs)):
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
                cur_loss = criterions[k](output, target)
                loss += cur_loss
                loss_dict[k].append(cur_loss.item())
            loss_dict["total_loss"].append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            mean_total_loss = np.mean(np.array(loss_dict["total_loss"]))
            print(f"\nTrain, iter: {iters}, total_loss: {mean_total_loss}", end=", ")
            writer.add_scalar("total_loss", mean_total_loss, iters)

            if iters % lr_decay_frequency == 0:
                for g in optimizer.param_groups:
                    g["lr"] *= 0.95

            if iters % save_frequency == 0 or iters == total_iters:
                torch.save(model.state_dict(), save_path)
                print("Model saved!")

            if iters == total_iters:
                break
            iters += 1

        if iters == total_iters:
            break

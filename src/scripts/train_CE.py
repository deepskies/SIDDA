import argparse
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from dataset import dataset_dict
from models import model_dict
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm


def set_all_seeds(num):
    random.seed(num)
    np.random.seed(num)
    torch.manual_seed(num)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(num)


def train_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    optimizer: optim.Optimizer,
    model_name: str,
    scheduler: optim.lr_scheduler = None,
    epochs: int = 100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_dir: str = "checkpoints",
    early_stopping_patience: int = 10,
    report_interval: int = 5,
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model.to(device)
    else:
        model.to(device)

    print("Model Loaded to Device!")
    best_val_acc, no_improvement_count = 0, 0
    losses, steps = [], []
    print("Training Started!")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for i, batch in tqdm(enumerate(train_dataloader)):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.float()

            optimizer.zero_grad()
            _, outputs = model(inputs)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            losses.append(loss.item())
            steps.append(epoch * len(train_dataloader) + i + 1)

        train_loss /= len(train_dataloader)
        print(f"Epoch: {epoch + 1}, Train Loss: {train_loss:.4e}")

        if scheduler is not None:
            scheduler.step()

        if (epoch + 1) % report_interval == 0:
            model.eval()
            correct, total, val_loss = 0, 0, 0.0

            with torch.no_grad():
                for batch in val_dataloader:
                    inputs, targets = batch
                    inputs, targets = inputs.to(device), targets.to(device)
                    inputs = inputs.float()
                    _, outputs = model(inputs)
                    loss = F.cross_entropy(outputs, targets)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()

            val_acc = 100 * correct / total
            val_loss /= len(val_dataloader)
            lr = (
                scheduler.get_last_lr()[0]
                if scheduler is not None
                else optimizer.param_groups[0]["lr"]
            )
            print(
                f"Epoch: {epoch + 1}, Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%, Learning rate: {lr}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improvement_count = 0
                best_val_epoch = epoch + 1
                if torch.cuda.device_count() > 1:
                    torch.save(
                        model.eval().module.state_dict(),
                        os.path.join(save_dir, "best_model.pt"),
                    )
                else:
                    torch.save(
                        model.eval().state_dict(),
                        os.path.join(save_dir, "best_model.pt"),
                    )
            else:
                no_improvement_count += 1

            if no_improvement_count >= early_stopping_patience:
                print(
                    f"Early stopping after {early_stopping_patience} epochs without improvement."
                )
                break

    if torch.cuda.device_count() > 1:
        torch.save(
            model.eval().module.state_dict(), os.path.join(save_dir, "final_model.pt")
        )
    else:
        torch.save(model.eval().state_dict(), os.path.join(save_dir, "final_model.pt"))
    np.save(os.path.join(save_dir, f"losses-{model_name}.npy"), np.array(losses))
    np.save(os.path.join(save_dir, f"steps-{model_name}.npy"), np.array(steps))

    # Plot loss vs. training step graph
    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses)
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Loss vs. Training Steps")
    plt.savefig(
        os.path.join(save_dir, "loss_vs_training_steps.png"), bbox_inches="tight"
    )

    return best_val_epoch, best_val_acc, losses[-1]


def main(config):
    model_name = str(config["model"]).strip()
    dataset_name = str(config["dataset"]).strip()
    model = model_dict[config["dataset"]][model_name]()

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(
        params_to_optimize,
        lr=config["parameters"]["lr"],
        weight_decay=config["parameters"]["weight_decay"],
    )

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config["parameters"]["milestones"],
        gamma=config["parameters"]["lr_decay"],
    )

    if dataset_name in ["shapes", "astro_objects"]:
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomRotation(180),
                transforms.Resize(100),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.Normalize(mean=(0.5,), std=(0.5,)),
            ]
        )

        val_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,)),
                transforms.Resize(100),
            ]
        )
    elif dataset_name == "mnist_m":
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomRotation(180),
                transforms.Resize(32),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

        val_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(32),
                transforms.Normalize(mean=(0.5,), std=(0.5,)),
            ]
        )

    elif dataset_name == "gz_evo":
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomRotation(180),
                transforms.Resize(100),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.RandomHorizontalFlip(p=0.3),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

        val_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(100),
                transforms.Normalize(mean=(0.5,), std=(0.5,)),
            ]
        )

    # Function to split dataset into train and validation subsets
    def split_dataset(dataset, val_size, train_transform, val_transform):
        val_size = int(len(dataset) * val_size)
        train_size = len(dataset) - val_size

        train_subset, val_subset = random_split(dataset, [train_size, val_size])

        # Apply transforms
        train_subset.dataset.transform = train_transform
        val_subset.dataset.transform = val_transform

        return train_subset, val_subset

    print("Loading datasets!")
    start = time.time()
    train_dataset = dataset_dict[dataset_name](
        input_path=config["train_data"]["input_path"],
        output_path=config["train_data"]["output_path"],
        transform=train_transform,
    )

    train_dataset, val_dataset = split_dataset(
        train_dataset,
        val_size=config["parameters"]["val_size"],
        train_transform=train_transform,
        val_transform=val_transform,
    )

    end = time.time()
    print(f"Datasets loaded and split in {end - start} seconds")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["parameters"]["batch_size"],
        shuffle=True,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config["parameters"]["batch_size"],
        shuffle=False,
        pin_memory=True,
    )

    timestr = time.strftime("%Y%m%d-%H%M%S")

    save_dir = config["save_dir"] + config["model"] + "_" + timestr
    best_val_epoch, best_val_acc, final_loss = train_model(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        model_name=model_name,
        scheduler=scheduler,
        epochs=config["parameters"]["epochs"],
        device=device,
        save_dir=save_dir,
        early_stopping_patience=config["parameters"]["early_stopping"],
        report_interval=config["parameters"]["report_interval"],
    )
    print("Training Done")
    config["best_val_acc"] = best_val_acc
    config["best_val_epoch"] = best_val_epoch
    config["final_loss"] = final_loss

    file = open(f"{save_dir}/config.yaml", "w")
    yaml.dump(config, file)
    file.close()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser(description="Train the models")
    parser.add_argument(
        "--config", metavar="config", required=True, help="Location of the config file"
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    set_all_seeds(config["seed"])

    main(config)

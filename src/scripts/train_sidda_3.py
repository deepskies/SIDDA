import argparse
import os
import random
import time

import geomloss
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


def kl_divergence(p, q):
    epsilon = 1e-6
    p = torch.clamp(p, min=epsilon)
    q = torch.clamp(q, min=epsilon)
    return torch.sum(p * torch.log(p / q), dim=-1)


def jensen_shannon_divergence(p, q):
    m = 0.5 * (p + q)
    jsd = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    return jsd


def jensen_shannon_distance(p, q):
    jsd = jensen_shannon_divergence(p, q)
    jsd = torch.clamp(jsd, min=0.0)
    return torch.sqrt(jsd)


def sinkhorn_loss(
    x,
    y,
    blur,
):
    loss = geomloss.SamplesLoss("sinkhorn", blur=blur, scaling=0.9, reach=None)
    return loss(x, y)


def set_all_seeds(num):
    random.seed(num)
    np.random.seed(num)
    torch.manual_seed(num)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(num)


def train_SIDDA(
    model: nn.Module,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    target_dataloader: DataLoader,
    target_val_dataloader: DataLoader,
    optimizer: optim.Optimizer,
    model_name: str,
    scheduler: optim.lr_scheduler = None,
    epochs: int = 100,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    save_dir: str = "checkpoints",
    early_stopping_patience: int = 10,
    report_interval: int = 1,
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model.to(device)
    else:
        model.to(device)

    warmup = config["parameters"]["warmup"]
    print("Model Loaded to Device!")
    best_val_acc, best_classification_loss, best_DA_loss, best_total_val_loss = (
        0,
        float("inf"),
        float("inf"),
        float("inf"),
    )
    no_improvement_count = 0
    losses, steps = [], []
    train_classification_losses, train_DA_losses = [], []
    val_losses, val_classification_losses, val_DA_losses = [], [], []
    max_distances, epoch_max_distances = [], []
    js_distances, epoch_js_distances = [], []
    blur_vals, epoch_blur_vals = [], []

    print("Training Started!")

    eta_1 = torch.nn.Parameter(torch.tensor(1.0, device=device))
    eta_2 = torch.nn.Parameter(torch.tensor(1.0, device=device))

    optimizer.add_param_group({"params": [eta_1, eta_2]})

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        classification_losses, DA_losses = [], []

        for i, (batch, target_batch) in tqdm(
            enumerate(zip(train_dataloader, target_dataloader))
        ):
            source_inputs, source_outputs = batch
            source_inputs, source_outputs = (
                source_inputs.to(device).float(),
                source_outputs.to(device),
            )

            target_inputs, _ = target_batch
            target_inputs = target_inputs.to(device).float()

            optimizer.zero_grad()

            if epoch < warmup:
                _, model_outputs = model(source_inputs)
                classification_loss = F.cross_entropy(model_outputs, source_outputs)
                loss = classification_loss
                DA_loss = None
            else:
                concatenated_inputs = torch.cat((source_inputs, target_inputs), dim=0)
                batch_size = source_inputs.size(0)

                features, model_outputs = model(concatenated_inputs)
                source_features = features[:batch_size]
                target_features = features[batch_size:]
                source_model_outputs = model_outputs[:batch_size]

                classification_loss = F.cross_entropy(
                    source_model_outputs, source_outputs
                )

                pairwise_distances = torch.cdist(source_features, target_features, p=2)
                flattened_distances = pairwise_distances.view(-1)
                max_distance = torch.max(flattened_distances)
                max_distances.append(max_distance.detach().cpu().numpy())
                js_distances.append(
                    jensen_shannon_distance(source_features, target_features)
                    .nanmean()
                    .item()
                )

                dynamic_blur_val = 0.05 * max_distance.detach().cpu().numpy()
                blur_vals.append(dynamic_blur_val)

                DA_loss = sinkhorn_loss(
                    source_features,
                    target_features,
                    blur=max(dynamic_blur_val, 0.01),  # Apply lower bound to blur
                )

                loss = (
                    (1 / (2 * eta_1**2)) * classification_loss
                    + (1 / (2 * eta_2**2)) * DA_loss
                    + torch.log(torch.abs(eta_1) * torch.abs(eta_2))
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            eta_1.data.clamp_(min=1e-3)
            eta_2.data.clamp_(min=0.25 * eta_1.data.item())
            optimizer.step()

            train_loss += loss.item()
            classification_losses.append(classification_loss.item())
            if epoch >= warmup:
                DA_losses.append(DA_loss.item())

        mean_max_distance = np.mean(max_distances)
        epoch_max_distances.append(mean_max_distance)

        mean_blur_val = np.mean(blur_vals)
        epoch_blur_vals.append(mean_blur_val)
        mean_js_distance = np.nanmean(js_distances)
        epoch_js_distances.append(mean_js_distance)

        train_loss /= len(train_dataloader)
        train_classification_loss = np.mean(classification_losses)
        train_DA_loss = np.mean(DA_losses) if DA_losses else None

        losses.append(train_loss)
        train_classification_losses.append(train_classification_loss)
        train_DA_losses.append(train_DA_loss)
        steps.append(epoch + 1)

        if epoch >= warmup:
            print(
                f"Epoch: {epoch + 1}, eta_1: {eta_1.item():.4f}, eta_2: {eta_2.item():.4f}"
            )
            print(f"Epoch: {epoch + 1}, Max Distance: {max_distance:.4f}")

        if epoch < warmup:
            print(f"Epoch: {epoch + 1}, Train Loss: {train_loss:.4e}")
            print(
                f"Epoch: {epoch + 1}, Classification Loss: {train_classification_loss:.4e}"
            )
        else:
            print(f"Epoch: {epoch + 1}, Train Loss: {train_loss:.4e}")
            print(
                f"Epoch: {epoch + 1}, Classification Loss: {train_classification_loss:.4e}, DA Loss: {train_DA_loss:.4e}"
            )

        if scheduler is not None:
            scheduler.step()

        if (epoch + 1) % report_interval == 0:
            model.eval()
            source_correct, target_correct, source_total, target_total, val_loss = (
                0,
                0,
                0,
                0,
                0.0,
            )
            val_classification_loss, val_DA_loss = 0.0, 0.0

            with torch.no_grad():
                for i, (batch, target_batch) in enumerate(
                    zip(val_dataloader, target_val_dataloader)
                ):
                    source_inputs, source_outputs = batch
                    source_inputs, source_outputs = (
                        source_inputs.to(device).float(),
                        source_outputs.to(device),
                    )
                    target_inputs, target_outputs = target_batch
                    target_inputs, target_outputs = (
                        target_inputs.to(device).float(),
                        target_outputs.to(device),
                    )

                    if epoch < warmup:
                        _, source_preds = model(source_inputs)
                        classification_loss_ = F.cross_entropy(
                            source_preds, source_outputs
                        )
                        combined_loss = classification_loss_
                        DA_loss_ = 0.0
                        target_preds = None

                    else:
                        concatenated_inputs = torch.cat(
                            (source_inputs, target_inputs), dim=0
                        )
                        batch_size = source_inputs.size(0)

                        features, preds = model(concatenated_inputs)
                        source_features = features[:batch_size]
                        target_features = features[batch_size:]
                        source_preds = preds[:batch_size]
                        target_preds = preds[batch_size:]

                        classification_loss_ = F.cross_entropy(
                            source_preds, source_outputs
                        )

                        pairwise_distances = torch.cdist(
                            source_features, target_features, p=2
                        )
                        flattened_distances = pairwise_distances.view(-1)
                        max_distance = torch.max(flattened_distances)

                        dynamic_blur_val = 0.05 * max_distance.detach().cpu().numpy()
                        DA_loss_ = sinkhorn_loss(
                            source_features,
                            target_features,
                            blur=max(dynamic_blur_val, 0.01),
                        )

                        combined_loss = classification_loss_ + DA_loss_

                        _, target_predicted = torch.max(target_preds.data, 1)
                        target_total += target_outputs.size(0)
                        target_correct += (
                            (target_predicted == target_outputs).sum().item()
                        )

                    val_loss += combined_loss.item()
                    val_classification_loss += classification_loss_.item()

                    if epoch >= warmup:
                        val_DA_loss += DA_loss_.item()

                    _, source_predicted = torch.max(source_preds.data, 1)
                    source_total += source_outputs.size(0)
                    source_correct += (source_predicted == source_outputs).sum().item()

            source_val_acc = 100 * source_correct / source_total

            if target_total > 0:
                target_val_acc = 100 * target_correct / target_total
            else:
                target_val_acc = 0.0

            val_loss /= len(val_dataloader)
            val_classification_loss /= len(val_dataloader)

            if epoch >= warmup:
                val_DA_loss /= len(val_dataloader)

            val_losses.append(val_loss)
            val_classification_losses.append(val_classification_loss)
            val_DA_losses.append(val_DA_loss)

            lr = (
                scheduler.get_last_lr()[0]
                if scheduler is not None
                else optimizer.param_groups[0]["lr"]
            )

            if epoch < warmup:
                print(
                    f"Epoch: {epoch + 1}, Total Validation Loss: {val_loss:.4f}, Source Validation Accuracy: {source_val_acc:.2f}%, Learning rate: {lr}"
                )
                print(
                    f"Epoch: {epoch + 1}, Validation Classification Loss: {val_classification_loss:.4e}"
                )
            else:
                print(
                    f"Epoch: {epoch + 1}, Total Validation Loss: {val_loss:.4f}, Source Validation Accuracy: {source_val_acc:.2f}%, Learning rate: {lr}, Target Validation Accuracy: {target_val_acc:.2f}%"
                )
                print(
                    f"Epoch: {epoch + 1}, Validation Classification Loss: {val_classification_loss:.4e}, Validation DA Loss: {val_DA_loss:.4e}"
                )

            if val_loss < best_total_val_loss and epoch >= warmup:
                best_total_val_loss = val_loss
                best_val_epoch = epoch + 1
                if torch.cuda.device_count() > 1:
                    torch.save(
                        model.eval().module.state_dict(),
                        os.path.join(save_dir, "best_model_total_val_loss.pt"),
                    )
                else:
                    torch.save(
                        model.eval().state_dict(),
                        os.path.join(save_dir, "best_model_total_val_loss.pt"),
                    )
                print(
                    f"Saved best total validation loss model at epoch {best_val_epoch}"
                )

            else:
                no_improvement_count += 1

            if source_val_acc >= best_val_acc:
                best_val_acc = source_val_acc
                best_val_acc_epoch = epoch + 1
                model_path = os.path.join(save_dir, "best_model_val_acc.pt")
                if torch.cuda.device_count() > 1:
                    torch.save(model.eval().module.state_dict(), model_path)
                else:
                    torch.save(model.eval().state_dict(), model_path)
                print(
                    f"Saved best validation accuracy model at epoch {best_val_acc_epoch}"
                )

            if val_classification_loss <= best_classification_loss and epoch >= warmup:
                best_classification_loss = val_classification_loss
                best_classification_loss_epoch = epoch + 1
                model_path = os.path.join(save_dir, "best_model_classification_loss.pt")
                if torch.cuda.device_count() > 1:
                    torch.save(model.eval().module.state_dict(), model_path)
                else:
                    torch.save(model.eval().state_dict(), model_path)
                print(
                    f"Saved lowest classification loss model at epoch {best_classification_loss_epoch}"
                )

            if val_DA_loss <= best_DA_loss and epoch >= warmup:
                best_DA_loss = val_DA_loss
                best_DA_epoch = epoch + 1
                model_path = os.path.join(save_dir, "best_model_DA_loss.pt")
                if torch.cuda.device_count() > 1:
                    torch.save(model.eval().module.state_dict(), model_path)
                else:
                    torch.save(model.eval().state_dict(), model_path)
                print(f"Saved lowest DA loss model at epoch {best_DA_epoch}")

            if no_improvement_count >= early_stopping_patience:
                print(
                    f"Early stopping after {early_stopping_patience} epochs without improvement in accuracy."
                )
                break

    if torch.cuda.device_count() > 1:
        torch.save(
            model.eval().module.state_dict(), os.path.join(save_dir, "final_model.pt")
        )
    else:
        torch.save(model.eval().state_dict(), os.path.join(save_dir, "final_model.pt"))

    loss_dir = os.path.join(save_dir, "losses")
    if not os.path.exists(loss_dir):
        os.makedirs(loss_dir)

    np.save(os.path.join(loss_dir, f"losses-{model_name}.npy"), np.array(losses))
    np.save(
        os.path.join(loss_dir, f"train_classification_losses-{model_name}.npy"),
        np.array(train_classification_losses),
    )
    np.save(
        os.path.join(loss_dir, f"train_DA_losses-{model_name}.npy"),
        np.array(train_DA_losses),
    )
    np.save(
        os.path.join(loss_dir, f"val_losses-{model_name}.npy"), np.array(val_losses)
    )
    np.save(
        os.path.join(loss_dir, f"val_classification_losses-{model_name}.npy"),
        np.array(val_classification_losses),
    )
    np.save(
        os.path.join(loss_dir, f"val_DA_losses-{model_name}.npy"),
        np.array(val_DA_losses),
    )
    np.save(os.path.join(loss_dir, f"steps-{model_name}.npy"), np.array(steps))
    np.save(
        os.path.join(loss_dir, f"max_distances-{model_name}.npy"),
        np.array(max_distances),
    )
    np.save(os.path.join(loss_dir, f"blur_vals-{model_name}.npy"), np.array(blur_vals))
    np.save(
        os.path.join(loss_dir, f"js_distances-{model_name}.npy"), np.array(js_distances)
    )
    np.save(
        os.path.join(loss_dir, f"epoch_max_distances-{model_name}.npy"),
        np.array(epoch_max_distances),
    )
    np.save(
        os.path.join(loss_dir, f"epoch_blur_vals-{model_name}.npy"),
        np.array(epoch_blur_vals),
    )
    np.save(
        os.path.join(loss_dir, f"epoch_js_distances-{model_name}.npy"),
        np.array(epoch_js_distances),
    )

    # Plotting the losses
    plt.figure(figsize=(14, 8))

    steps = np.array(steps)
    validation_steps = steps[::report_interval]
    losses = np.array(losses)
    train_classification_losses = np.array(train_classification_losses)
    train_DA_losses = np.array(train_DA_losses)
    val_losses = np.array(val_losses)
    val_classification_losses = np.array(val_classification_losses)
    val_DA_losses = np.array(val_DA_losses)

    # Plot Training Losses
    plt.subplot(2, 1, 1)
    plt.plot(steps, losses, label="Train Total Loss")
    plt.plot(steps, train_classification_losses, label="Train Classification Loss")
    plt.plot(steps, train_DA_losses, label="Train DA Loss")
    plt.axvline(x=best_val_epoch, color="b", linestyle="--", label="Best Val Epoch")
    plt.axvline(
        x=best_classification_loss_epoch,
        color="y",
        linestyle="--",
        label="Best Classification Epoch",
    )
    plt.axvline(x=best_DA_epoch, color="g", linestyle="--", label="Best DA Epoch")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.yscale("log")
    plt.legend()

    # Plot Validation Losses
    plt.subplot(2, 1, 2)
    plt.plot(validation_steps, val_losses, label="Validation Total Loss")
    plt.plot(
        validation_steps,
        val_classification_losses,
        label="Validation Classification Loss",
    )
    plt.plot(validation_steps, val_DA_losses, label="Validation DA Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Validation Losses")
    plt.yscale("log")
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(loss_dir, f"losses_plot-{model_name}.png"))
    plt.close()

    plt.figure(figsize=(10, 5))

    plt.plot(steps, epoch_max_distances)
    plt.axvline(x=best_val_epoch, color="b", linestyle="--", label="Best Val Epoch")
    plt.axvline(
        x=best_classification_loss_epoch,
        color="y",
        linestyle="--",
        label="Best Classification Epoch",
    )
    plt.axvline(x=best_DA_epoch, color="g", linestyle="--", label="Best DA Epoch")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Max Distance")
    plt.title("Max Distance vs. Training Steps")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(loss_dir, f"max_distance_plot-{model_name}.png"))
    plt.close()

    plt.figure(figsize=(10, 5))

    plt.plot(steps, epoch_blur_vals)
    plt.axhline(y=0.01, color="r", linestyle="--")
    plt.axhline(y=0.05, color="g", linestyle="--")
    plt.axvline(x=best_val_epoch, color="b", linestyle="--", label="Best Val Epoch")
    plt.axvline(
        x=best_classification_loss_epoch,
        color="y",
        linestyle="--",
        label="Best Classification Epoch",
    )
    plt.axvline(x=best_DA_epoch, color="g", linestyle="--", label="Best DA Epoch")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Blur Value")
    plt.title("Blur Value vs. Training Steps")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(loss_dir, f"blur_value_plot-{model_name}.png"))
    plt.close()

    plt.figure(figsize=(10, 5))

    plt.plot(steps, epoch_js_distances)
    plt.axvline(x=best_val_epoch, color="b", linestyle="--", label="Best Val Epoch")
    plt.axvline(
        x=best_classification_loss_epoch,
        color="y",
        linestyle="--",
        label="Best Classification Epoch",
    )
    plt.axvline(x=best_DA_epoch, color="g", linestyle="--", label="Best DA Epoch")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("JS Distance")
    plt.title("JS Distance vs. Training Steps")
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(os.path.join(loss_dir, f"js_distance_plot-{model_name}.png"))
    plt.close()

    return (
        best_val_epoch,
        best_val_acc,
        best_classification_loss_epoch,
        best_classification_loss,
        best_DA_epoch,
        best_DA_loss,
        losses[-1],
    )


def main(config):
    model_name = str(config["model"]).strip()
    dataset_name = str(config["dataset"]).strip()
    model = model_dict["shapes"][model_name]()

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

    # Load source dataset
    train_dataset = dataset_dict[dataset_name](
        input_path=config["train_data"]["input_path"],
        output_path=config["train_data"]["output_path"],
        transform=train_transform,
    )

    # Split source dataset into train and validation sets
    train_dataset, val_dataset = split_dataset(
        train_dataset,
        val_size=config["parameters"]["val_size"],
        train_transform=train_transform,
        val_transform=val_transform,
    )

    # Load target dataset
    target_dataset = dataset_dict[dataset_name](
        input_path=config["train_data"]["target_input_path"],
        output_path=config["train_data"][
            "target_output_path"
        ],  ## outputs dont get used in training
        transform=train_transform,
    )

    # Split target dataset into train and validation sets
    target_dataset, val_target_dataset = split_dataset(
        target_dataset,
        val_size=config["parameters"]["val_size"],
        train_transform=train_transform,
        val_transform=val_transform,
    )

    end = time.time()
    print(f"Datasets loaded and split in {end - start} seconds")

    # Dataloaders can be created if needed
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

    target_dataloader = DataLoader(
        target_dataset,
        batch_size=config["parameters"]["batch_size"],
        shuffle=True,
        pin_memory=True,
    )
    target_val_dataloader = DataLoader(
        val_target_dataset,
        batch_size=config["parameters"]["batch_size"],
        shuffle=False,
        pin_memory=True,
    )

    timestr = time.strftime("%Y%m%d-%H%M%S")

    save_dir = config["save_dir"] + config["model"] + "_DA_" + timestr
    (
        best_val_epoch,
        best_val_acc,
        best_classification_epoch,
        best_classification_loss,
        best_DA_epoch,
        best_DA_loss,
        final_loss,
    ) = train_SIDDA(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        target_dataloader=target_dataloader,
        target_val_dataloader=target_val_dataloader,
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
    config["final_loss"] = float(final_loss)
    config["best_classification_epoch"] = best_classification_epoch
    config["best_classification_loss"] = best_classification_loss
    config["best_DA_epoch"] = best_DA_epoch
    config["best_DA_loss"] = best_DA_loss

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

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import torch
import torch.nn as nn
import yaml
from dataset import classes_dict, dataset_dict
from models import model_dict
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_models(directory_path: str, 
                model_name: str,
                dataset_name: str) -> list:
    """Load models from a directory

    Args:
        directory_path (str): directory with the trained models
        model_name (str): name of the model to be loaded (following the model_dict)

    Returns:
        loaded models (list): list of loaded models
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    models = []

    for file_name in os.listdir(directory_path):
        if file_name.endswith(".pt"):
            file_path = os.path.join(directory_path, file_name)
            print(f"Loading {model_name} from {file_path}...")
            model = model_dict[dataset_name][model_name]()
            model.eval()
            model.load_state_dict(torch.load(file_path, map_location=device))

            model_name_no_ext = file_name[:-3]
            models.append((model, model_name_no_ext))
            print(f"Finished Loading {model_name} from {file_path}")

    if not models:
        print(
            f"No models containing 'best_model' ending with '.pt' found in {directory_path}."
        )

    return models


@torch.no_grad()
def compute_metrics(
    test_loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    model_name: str,
    save_dir: str,
    output_name: str,
    classes: tuple,
):
    """Compute metrics for the model with multi-layer latents.

    Returns:
        sklearn_report (dict)
    """
    device = next(model.parameters()).device
    y_pred, y_true = [], []
    latent_batches = []

    # support multi-gpu
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model.eval()

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, unit="batch"):
            # send to device
            inputs = inputs.to(device).float()
            labels = labels.to(device)

            # forward
            out = model(inputs)
            feats = out['features']    # list of 4 tensors shape (B, Dₗ)
            logits = out['logits']     # (B, num_classes)

            # prediction
            preds = logits.argmax(dim=1)
            y_pred.extend(preds.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

            # concatenate layer-features → one vector per sample
            # feats = [ (B,8), (B,16), (B,32), (B,256) ] → cat → (B,312)
            cat_feats = torch.cat(feats, dim=1)
            latent_batches.append(cat_feats.cpu().numpy())

    # stack all batches into (N, 312) array
    latent_vectors = np.vstack(latent_batches)

    # save latents
    features_dir = os.path.join(save_dir, "latent_vectors")
    os.makedirs(features_dir, exist_ok=True)
    np.save(
        os.path.join(features_dir, f"latents_{model_name}_{output_name}.npy"),
        latent_vectors
    )

    # save predictions
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)
    y_pred_dir = os.path.join(save_dir, "y_pred")
    os.makedirs(y_pred_dir, exist_ok=True)
    np.save(
        os.path.join(y_pred_dir, f"y_pred_{model_name}_{output_name}.npy"),
        y_pred
    )

    # classification report
    sklearn_report = classification_report(
        y_true, y_pred, target_names=classes, output_dict=True
    )

    # normalized confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(
        cm / cm.sum(axis=1, keepdims=True),
        index=classes, columns=classes
    )
    cm_dir = os.path.join(save_dir, "confusion_matrix")
    os.makedirs(cm_dir, exist_ok=True)
    plt.figure(figsize=(12,7))
    sn.heatmap(df_cm, annot=True, fmt=".2f")
    plt.title(f"{model_name} Confusion Matrix")
    plt.savefig(
        os.path.join(cm_dir, f"confusion_matrix_{model_name}_{output_name}.png"),
        bbox_inches="tight"
    )
    plt.close()

    return sklearn_report


@torch.no_grad()
def main(
    model_dir: str,
    output_name: str,
    x_test_path: str,
    y_test_path: str,
    model_name: str,
    classes: tuple,
    dataset: str,
):
    metrics_dir = os.path.join(model_dir, "metrics")
    if not os.path.exists(metrics_dir):
        os.makedirs(metrics_dir)

    if dataset in ["shapes", "astronomical_objects"]:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5,), std=(0.5,)),
                transforms.Resize(100),
            ]
        )

    elif dataset == "mnist_m":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                transforms.Resize(32),
            ]
        )

    elif dataset == "gz_evo":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                transforms.Resize(100),
            ]
        )
        
    elif dataset == "mrssc2":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(100),
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

    test_dataset = dataset_dict[dataset](x_test_path, y_test_path, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)

    models = load_models(model_dir, model_name, dataset)
    if not models:
        print("Models could not be loaded.")
        return

    for model, model_file_name in models:
        model_metrics = {
            class_name: {"precision": [], "recall": [], "f1-score": [], "support": []}
            for class_name in classes
        }
        model_metrics["accuracy"] = []
        model_metrics["macro avg"] = {
            "precision": [],
            "recall": [],
            "f1-score": [],
            "support": [],
        }
        model_metrics["weighted avg"] = {
            "precision": [],
            "recall": [],
            "f1-score": [],
            "support": [],
        }

        full_report = compute_metrics(
            test_loader=test_dataloader,
            model=model,
            model_name=model_name,
            save_dir=model_dir,
            output_name=f"{output_name}_{model_file_name}",
            classes=classes,
        )
        model_metrics = full_report

        print("Compiling Metrics")
        output_file_name = f"{output_name}_{model_file_name}.yaml"
        with open(os.path.join(metrics_dir, output_file_name), "w") as file:
            yaml.dump(model_metrics, file)

        print(f"Metrics saved at {os.path.join(model_dir, output_file_name)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test models")
    parser.add_argument(
        "--dataset",
        type=str,
        default="gz_evo",
        help="Dataset to be used for evaluation",
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the trained models"
    )
    parser.add_argument(
        "--x_test_path", type=str, required=True, help="Path to the x_test data"
    )
    parser.add_argument(
        "--y_test_path", type=str, required=True, help="Path to the y_test data"
    )
    parser.add_argument(
        "--output_name",
        type=str,
        required=True,
        help="Name of the output file for the results",
    )
    parser.add_argument(
        "--model_name", type=str, help="Name of the model to be evaluated"
    )
    
    args = parser.parse_args()

    main(
        model_dir=args.model_path,
        output_name=args.output_name,
        x_test_path=args.x_test_path,
        y_test_path=args.y_test_path,
        model_name=args.model_name,
        classes=classes_dict[args.dataset],
        dataset = args.dataset
    )

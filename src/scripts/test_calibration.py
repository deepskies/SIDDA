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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, classification_report, confusion_matrix
from test import load_models
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to compute ECE
def expected_calibration_error(
    y_true: np.ndarray, y_probs: np.ndarray, num_bins: int = 10
) -> float:
    """Compute the Expected Calibration Error (ECE) for multi-class classification."""
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    ece = 0.0
    total_samples = len(y_true)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        bin_size = 0
        bin_error = 0.0

        for i in range(total_samples):
            prob_pred = y_probs[i, np.argmax(y_probs[i])]

            if bin_lower < prob_pred <= bin_upper:
                bin_size += 1
                is_correct = y_true[i] == np.argmax(y_probs[i])
                bin_error += np.abs(prob_pred - is_correct)

        if bin_size > 0:
            ece += bin_error / total_samples

    return ece

@torch.no_grad()
def compute_metrics_with_calibration(
    test_loader: DataLoader,
    model: nn.Module,
    model_name: str,
    save_dir: str,
    output_name: str,
    classes: tuple,
) -> tuple:
    """Compute metrics for a model with calibration

    Args:
        test_loader (nn.DataLoader): test data loader
        model (nn.Module): model to be evaluated
        model_name (str): name of the model
        save_dir (str): directory to save the results
        output_name (str): name of the output file
        classes (list): list of classes

    Returns:
        _type_: _description_
    """
    y_pred, y_true, feature_maps, y_proba = [], [], [], []
    model.to(device)
    model.eval()

    for batch in tqdm(test_loader, unit="batch", total=len(test_loader)):
        input, output = batch
        input, output = input.to(device).float(), output.to(device)
        features, preds = model(input)
        probs = torch.softmax(preds, dim=1)
        feature_maps.extend(features.cpu().numpy())
        y_pred.extend(torch.argmax(probs, dim=1).cpu().numpy())
        y_proba.extend(probs.cpu().numpy())
        y_true.extend(output.cpu().numpy())

    y_pred, y_true = np.asarray(y_pred), np.asarray(y_true)
    feature_maps = np.asarray(feature_maps)
    flattened_features = feature_maps.reshape(feature_maps.shape[0], -1)

    features_dir = os.path.join(save_dir, "features")
    os.makedirs(features_dir, exist_ok=True)
    np.save(
        f"{features_dir}/features_{model_name}_{output_name}.npy", flattened_features
    )

    print("Calibrating classification scores...")
    calibrator = CalibratedClassifierCV(
        estimator=LogisticRegression(), method="sigmoid"
    )
    calibrator.fit(flattened_features, y_true)
    calibrated_proba = calibrator.predict_proba(flattened_features)

    proba_dir = os.path.join(save_dir, "calibrated_probs")
    os.makedirs(proba_dir, exist_ok=True)
    np.save(
        f"{proba_dir}/calibrated_probs_{model_name}_{output_name}.npy", calibrated_proba
    )

    y_pred_calibrated = np.argmax(calibrated_proba, axis=1)
    sklearn_report = classification_report(
        y_true, y_pred_calibrated, output_dict=True, target_names=classes
    )

    cf_matrix = confusion_matrix(y_true, y_pred_calibrated)
    df_cm = pd.DataFrame(
        cf_matrix / np.sum(cf_matrix, axis=1)[:, None],
        index=[i for i in classes],
        columns=[i for i in classes],
    )
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.title(f"{model_name} Calibrated Confusion Matrix")
    plt.savefig(
        os.path.join(
            save_dir, f"confusion_matrix/confusion_matrix_calibrated_{model_name}_{output_name}.png"
        ),
        bbox_inches="tight",
    )
    plt.close()

    brier_scores = [
        brier_score_loss(y_true == i, calibrated_proba[:, i])
        for i in range(calibrated_proba.shape[1])
    ]
    mean_brier_score = float(np.mean(brier_scores))

    ece = float(expected_calibration_error(y_true, calibrated_proba))

    return sklearn_report, ece, mean_brier_score


@torch.no_grad()
def main(model_dir: str,
         output_name: str,
         x_test_path: str,
         y_test_path: str,
         model_name: str,
         classes: list,
         dataset: str
    ) -> None:
    """Main function to evaluate models with calibration

    Args:
        model_dir (str): directory containing the models
        output_name (str): name of the output file
        x_test_path (str): path to the test images
        y_test_path (str): path to the test labels
        model_name (str): name of the model
        classes (list): list of classes
    """
    metrics_dir = os.path.join(model_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

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

    test_dataset = dataset_dict[dataset](
        x_test_path, y_test_path, transform=transform
    )
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)

    models = load_models(model_dir, model_name, dataset)
    if not models:
        print("Models could not be loaded.")
        return

    for model, model_file_name in models:
        model_metrics, ece, brier_score = compute_metrics_with_calibration(
            test_loader=test_dataloader,
            model=model,
            model_name=model_file_name,
            save_dir=model_dir,
            output_name=output_name,
            classes=classes,
        )

        # Add ECE and Brier score to the metrics
        model_metrics["ECE"] = ece
        model_metrics["Brier Score"] = brier_score

        print("Compiling Metrics")
        output_file_name = f"{output_name}_{model_file_name}.yaml"
        with open(os.path.join(metrics_dir, output_file_name), "w") as file:
            yaml.dump(model_metrics, file)

        print(f"Metrics saved at {os.path.join(metrics_dir, output_file_name)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate models with calibration")
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

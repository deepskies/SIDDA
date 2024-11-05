import os
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from toy_model_simple import mnistm_models
from toy_dataset import MnistM
from utils import OnePixelAttack

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classes = (
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
)


def load_models(directory_path, model_name):
    models = []
    
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.pt'):
            file_path = os.path.join(directory_path, file_name)
            print(f'Loading {model_name} from {file_path}...')
            model = mnistm_models[model_name](num_classes=10)
            model.eval()
            model.load_state_dict(torch.load(file_path, map_location=device))
            
            model_name_no_ext = file_name[:-3]
            models.append((model, model_name_no_ext))
            print(f'Finished Loading {model_name} from {file_path}')
    
    if not models:
        print(f"No models containing 'best_model' ending with '.pt' found in {directory_path}.")
    
    return models


@torch.no_grad()
def compute_metrics_with_calibration(test_loader, model, model_name, save_dir, output_name):
    y_pred, y_true, feature_maps, y_proba = [], [], [], []
    model.to(device)
    model.eval()

    for batch in tqdm(test_loader, unit="batch", total=len(test_loader)):
        input, output = batch
        input, label = input.to(device).float(), output.to(device)
        features, preds = model(input)
        probs = torch.softmax(preds, dim=1)
        feature_maps.extend(features.cpu().numpy())
        
        y_pred.extend(torch.argmax(probs, dim=1).cpu().numpy())
        y_proba.extend(probs.cpu().numpy())
        y_true.extend(output.cpu().numpy())
    
    y_pred, y_true = np.asarray(y_pred), np.asarray(y_true)
    feature_maps = np.asarray(feature_maps)
    flattened_features = feature_maps.reshape(feature_maps.shape[0], -1)
    
    # Save features and predictions
    features_dir = os.path.join(save_dir, 'features')
    os.makedirs(features_dir, exist_ok=True)
    np.save(f"{features_dir}/features_{model_name}_{output_name}.npy", flattened_features)
    
    # Calibrate probabilities using sklearn
    print('Calibrating classification scores...')
    calibrator = CalibratedClassifierCV(base_estimator=LogisticRegression(), method='sigmoid')
    calibrator.fit(flattened_features, y_true)
    calibrated_proba = calibrator.predict_proba(flattened_features)
    
    # Save the calibrated probabilities
    proba_dir = os.path.join(save_dir, 'calibrated_proba')
    os.makedirs(proba_dir, exist_ok=True)
    np.save(f"{proba_dir}/calibrated_proba_{model_name}_{output_name}.npy", calibrated_proba)
    
    # Compute the classification report and confusion matrix with calibrated probabilities
    y_pred_calibrated = np.argmax(calibrated_proba, axis=1)
    sklearn_report = classification_report(y_true, y_pred_calibrated, output_dict=True, target_names=classes)
    
    # Confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred_calibrated)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes], columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.title(f'{model_name} Calibrated Confusion Matrix')
    plt.savefig(os.path.join(save_dir, f"confusion_matrix_calibrated_{model_name}_{output_name}.png"), bbox_inches='tight')
    plt.close()
    
    return sklearn_report


@torch.no_grad()
def main(model_dir, output_name, x_test_path, y_test_path, model_name, N=None, adversarial_attack=False):
    metrics_dir = os.path.join(model_dir, 'metrics')
    os.makedirs(metrics_dir, exist_ok=True)
    
    if adversarial_attack:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(32),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            OnePixelAttack()
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            transforms.Resize(32)
        ])

    test_dataset = MnistM(x_test_path, y_test_path, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)

    models = load_models(model_dir, model_name)
    if not models:
        print("Models could not be loaded.")
        return
    
    for model, model_file_name in models:
        model_metrics = compute_metrics_with_calibration(test_loader=test_dataloader, model=model, 
                                                         model_name=model_name, save_dir=model_dir, 
                                                         output_name=output_name)
        print('Compiling Metrics')
        output_file_name = f'{output_name}_{model_file_name}.yaml'
        with open(os.path.join(metrics_dir, output_file_name), 'w') as file:
            yaml.dump(model_metrics, file)

        print(f'Metrics saved at {os.path.join(metrics_dir, output_file_name)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate models with calibration')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained models')
    parser.add_argument('--x_test_path', type=str, required=True, help='Path to the x_test data')
    parser.add_argument('--y_test_path', type=str, required=True, help='Path to the y_test data')
    parser.add_argument('--output_name', type=str, required=True, help='Name of the output file for the results')
    parser.add_argument('--model_name', type=str, help='Name of the model to be evaluated')
    parser.add_argument('--adversarial_attack', action='store_true', help='Apply adversarial attack to the input data')
    args = parser.parse_args()
    
    main(model_dir=args.model_path, output_name=args.output_name, x_test_path=args.x_test_path, y_test_path=args.y_test_path, adversarial_attack=args.adversarial_attack,
         model_name=args.model_name)

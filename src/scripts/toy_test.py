import os
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from toy_models import d4_model
from toy_dataset import Shapes
from utils import OnePixelAttack
from sklearn.manifold import Isomap

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

classes = (
    'lines', 'rectangles', 'circles'
)


def load_models(directory_path, model_name='D4'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    models = []
    
    # Search for files that contain 'best_model' in their names
    for file_name in os.listdir(directory_path):
        if 'best_model' in file_name:
            file_path = os.path.join(directory_path, file_name)
            print(f'Loading {model_name} from {file_path}...')
            
            model = d4_model()
            model.eval()
            model.load_state_dict(torch.load(file_path, map_location=device))
            
            models.append((model, file_name))
            print(f'Finished Loading {model_name} from {file_path}')
    
    if not models:
        print(f"No models containing 'best_model' found in {directory_path}.")
    
    return models



@torch.no_grad()
def compute_metrics(test_loader, model, model_name, save_dir, output_name):
    y_pred, y_true, feature_maps = [], [], []
    model = nn.DataParallel(model)
    model.to(device)
    model.eval()

    for batch in tqdm(test_loader, unit="batch", total=len(test_loader)):
        input, label = batch
        input, label = input.to(device), label.to(device)
        input = input.float()
        features, outputs = model(input)
        pred_labels = torch.argmax(outputs, dim=-1).cpu().numpy()
        feature_maps.extend(features.cpu().numpy())
        
        y_pred.extend(pred_labels)
        y_true.extend(label.cpu().numpy())
    
    y_pred, y_true = np.asarray(y_pred), np.asarray(y_true)
    feature_maps = np.asarray(feature_maps)
    flattened_features = feature_maps.reshape(feature_maps.shape[0], -1)
    
    isomap = Isomap(n_components=2, n_neighbors=5)
    isomap_embedding = isomap.fit_transform(flattened_features)
    plt.scatter(isomap_embedding[:, 0], isomap_embedding[:, 1], c=y_pred, cmap='viridis')
    plt.colorbar()
    plt.savefig(os.path.join(save_dir, f"isomap_{model_name}_{output_name}.png"), bbox_inches='tight')
    np.save(f"{save_dir}/isomap_{model_name}_{output_name}.npy", isomap_embedding)
    plt.close()
  
    sklearn_report = classification_report(y_true, y_pred, output_dict=True, target_names=classes)

    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes], columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.title(f'{model_name} Confusion Matrix')
    plt.savefig(os.path.join(save_dir, f"confusion_matrix_{model_name}_{output_name}.png"), bbox_inches='tight')
    plt.close()
    
    return sklearn_report

@torch.no_grad()
def main(model_dir, output_name, x_test_path, y_test_path, N=None, adversarial_attack=False):
    if adversarial_attack:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(100),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
            OnePixelAttack()
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,)),
            transforms.Resize(100)
        ])

    test_dataset = Shapes(x_test_path, y_test_path, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)

    model_name = 'D4'
    models = load_models(model_dir, model_name)
    if not models:
        print(f"Models could not be loaded.")
        return
    
    for model, model_file_name in models:
        model_metrics = {class_name: {"precision": [], "recall": [], "f1-score": [], "support": []} 
                         for class_name in classes}
        model_metrics['accuracy'] = []
        model_metrics['macro avg'] = {"precision": [], "recall": [], "f1-score": [], "support": []}
        model_metrics['weighted avg'] = {"precision": [], "recall": [], "f1-score": [], "support": []}

        if N is not None:
            for i in range(N):
                print(f"Starting evaluation {i + 1} of {N}")
                full_report = compute_metrics(test_loader=test_dataloader, model=model, 
                                              model_name=f"{model_name}_{i + 1}", save_dir=model_dir, output_name=output_name)
                
                # Append the metrics of this iteration to the respective lists
                for class_name in classes:
                    for metric in ["precision", "recall", "f1-score", "support"]:
                        model_metrics[class_name][metric].append(float(full_report[class_name][metric]))
                        
                # Append accuracy, macro avg, and weighted avg
                model_metrics['accuracy'].append(float(full_report['accuracy']))
                for metric in ["precision", "recall", "f1-score", "support"]:
                    model_metrics['macro avg'][metric].append(float(full_report['macro avg'][metric]))
                    model_metrics['weighted avg'][metric].append(float(full_report['weighted avg'][metric]))
            
            # Compute the mean of the metrics across all iterations
            for class_name in classes:
                for metric in ["precision", "recall", "f1-score", "support"]:
                    model_metrics[class_name][metric] = float(np.mean(model_metrics[class_name][metric]))

            # Compute the mean of accuracy, macro avg, and weighted avg
            model_metrics['accuracy'] = float(np.mean(model_metrics['accuracy']))
            for metric in ["precision", "recall", "f1-score", "support"]:
                model_metrics['macro avg'][metric] = float(np.mean(model_metrics['macro avg'][metric]))
                model_metrics['weighted avg'][metric] = float(np.mean(model_metrics['weighted avg'][metric]))

        else:
            full_report = compute_metrics(test_loader=test_dataloader, model=model, 
                                          model_name=model_name, save_dir=model_dir, 
                                          output_name=f"{output_name}_{model_file_name}")
            model_metrics = full_report

        print('Compiling Metrics')
        output_file_name = f'{output_name}_{model_file_name}.yaml'
        with open(os.path.join(model_dir, output_file_name), 'w') as file:
            yaml.dump(model_metrics, file)

        print(f'Metrics saved at {os.path.join(model_dir, output_file_name)}')
        
        del model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Galaxy10 models')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained models')
    parser.add_argument('--x_test_path', type=str, required=True, help='Path to the x_test data')
    parser.add_argument('--y_test_path', type=str, required=True, help='Path to the y_test data')
    parser.add_argument('--output_name', type=str, required=True, help='Name of the output file for the results')
    parser.add_argument('--adversarial_attack', action='store_true', help='Apply adversarial attack to the input data')
    args = parser.parse_args()
    
    main(model_dir=args.model_path, output_name=args.output_name, x_test_path=args.x_test_path, y_test_path=args.y_test_path, adversarial_attack=args.adversarial_attack)

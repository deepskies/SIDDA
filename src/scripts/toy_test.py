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

device = ('cuda' if torch.cuda.is_available() else 'cpu')

classes = (
    'lines', 'rectangles', 'circles'
)

def load_model(checkpoint_path):
    model = d4_model()
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

@torch.no_grad()
def compute_metrics(test_loader, model, model_name, save_dir, output_name):
    y_pred, y_true = [], []
    model = nn.DataParallel(model)
    model.to(device)
    model.eval()

    for batch in tqdm(test_loader, unit="batch", total=len(test_loader)):
        input, label, _, _ = batch
        input, label = input.to(device), label.to(device)
        outputs = model(input)
        pred_labels = torch.argmax(outputs, dim=-1).cpu().numpy()
        
        y_pred.extend(pred_labels)
        y_true.extend(label.cpu().numpy())
    
    y_pred, y_true = np.asarray(y_pred), np.asarray(y_true)  
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
            transforms.Resize(255),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            OnePixelAttack()
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, ), std=(0.5, )),
        ])

    test_dataset = Shapes(x_test_path, y_test_path, transform)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)

    if N is not None:
        trained_models = load_model(model_dir)
        print('Model Loaded!')
        
        model_metrics = {model_name: {class_name: {"precision": [], "recall": [], "f1-score": [], "support": []} 
                                      for class_name in classes} 
                         for model_name in trained_models.keys()}
        
        
        for model_name in model_metrics:
            model_metrics[model_name]['accuracy'] = []
            model_metrics[model_name]['macro avg'] = {"precision": [], "recall": [], "f1-score": [], "support": []}
            model_metrics[model_name]['weighted avg'] = {"precision": [], "recall": [], "f1-score": [], "support": []}
    

        for i in range(N):
            print(f"Starting evaluation {i + 1} of {N}")
            for model_name, model in tqdm(trained_models.items()):
                full_report = compute_metrics(test_loader=test_dataloader, model=model, 
                                              model_name=f"{model_name}_{i + 1}", save_dir=model_dir, output_name=output_name)
                
                # Append the metrics of this iteration to the respective lists
                for class_name in classes:
                    for metric in ["precision", "recall", "f1-score", "support"]:
                        model_metrics[model_name][class_name][metric].append(float(full_report[class_name][metric]))
                        
                # Append accuracy, macro avg, and weighted avg
                model_metrics[model_name]['accuracy'].append(float(full_report['accuracy']))
                for metric in ["precision", "recall", "f1-score", "support"]:
                    model_metrics[model_name]['macro avg'][metric].append(float(full_report['macro avg'][metric]))
                    model_metrics[model_name]['weighted avg'][metric].append(float(full_report['weighted avg'][metric]))
    
        # Compute the mean of the metrics across all iterations
        for model_name in model_metrics:
            for class_name in classes:
                for metric in ["precision", "recall", "f1-score", "support"]:
                    model_metrics[model_name][class_name][metric] = float(np.mean(model_metrics[model_name][class_name][metric]))

            # Compute the mean of accuracy, macro avg, and weighted avg
            model_metrics[model_name]['accuracy'] = float(np.mean(model_metrics[model_name]['accuracy']))
            for metric in ["precision", "recall", "f1-score", "support"]:
                model_metrics[model_name]['macro avg'][metric] = float(np.mean(model_metrics[model_name]['macro avg'][metric]))
                model_metrics[model_name]['weighted avg'][metric] = float(np.mean(model_metrics[model_name]['weighted avg'][metric]))

        print('Compiling All Metrics')
        with open(f'{model_dir}/{output_name}.yaml', 'w') as file:
            yaml.dump(model_metrics, file)
        
    else:
        trained_models = load_model(model_dir)
        print('All Models Loaded!')
        
        model_metrics = dict.fromkeys(trained_models.keys())
        
        for model_name, model in tqdm(trained_models.items()):
            full_report = compute_metrics(test_loader=test_dataloader, model=model, model_name=model_name, save_dir=model_dir, output_name=output_name)
            model_metrics[model_name] = full_report

        print('Compiling All Metrics')
        with open(f'{model_dir}/{output_name}.yaml', 'w') as file:
            yaml.dump(model_metrics, file)

    print(f'Metrics saved at {model_dir}/{output_name}.yaml')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Galaxy10 models')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained models')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the test data file')
    parser.add_argument('--output_name', type=str, required=True, help='Name of the output file for the results')
    parser.add_argument('--adversarial_attack', action='store_true', help='Apply adversarial attack to the input data')
    args = parser.parse_args()
    
    main(model_dir=args.model_path, output_name=args.output_name, data_path=args.data_path, adversarial_attack=args.adversarial_attack)

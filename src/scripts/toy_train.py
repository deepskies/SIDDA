import os
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.utils.data import DataLoader, random_split
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torchvision import transforms
# from toy_models import d4_model, feature_fields
from toy_model_simple import cnn, d4_model, mmnistm_models
from toy_dataset import Shapes, Blobs, MnistM
from tqdm import tqdm
import random
import geomloss

import torch
import torch.nn.functional as F

def kl_divergence(p, q):
    epsilon = 1e-6  # Larger epsilon to avoid numerical issues
    p = torch.clamp(p, min=epsilon)
    q = torch.clamp(q, min=epsilon)
    return torch.sum(p * torch.log(p / q), dim=-1)

def jensen_shannon_divergence(p, q):
    m = 0.5 * (p + q)
    jsd = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    return jsd

def jensen_shannon_distance(p, q):
    jsd = jensen_shannon_divergence(p, q)
    jsd = torch.clamp(jsd, min=0.0)  # Ensure no negative values
    return torch.sqrt(jsd)


def sinkhorn_loss(x, 
                  y,
                  blur,
                  scaling,
                  reach
            ):
    
    loss = geomloss.SamplesLoss(loss=config['DA_metric'], 
                                blur = blur, 
                                scaling = scaling, 
                                reach = reach
                            )
    return loss(x, y)


def set_all_seeds(num):
    random.seed(num)
    np.random.seed(num)
    torch.manual_seed(num)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(num)

def train_model(model, 
                train_dataloader, 
                val_dataloader, 
                optimizer, 
                model_name, 
                scheduler = None, 
                epochs=100, 
                device='cuda' if torch.cuda.is_available() else 'mps',
                save_dir='checkpoints', 
                early_stopping_patience=10, 
                report_interval=5
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
            lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']
            print(f"Epoch: {epoch + 1}, Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%, Learning rate: {lr}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improvement_count = 0
                best_val_epoch = epoch + 1
                if torch.cuda.device_count() > 1:
                    torch.save(model.eval().module.state_dict(), os.path.join(save_dir, "best_model.pt"))
                else:
                    torch.save(model.eval().state_dict(), os.path.join(save_dir, "best_model.pt"))
            else:
                no_improvement_count += 1

            if no_improvement_count >= early_stopping_patience:
                print(f"Early stopping after {early_stopping_patience} epochs without improvement.")
                break
    
    if torch.cuda.device_count() > 1:
        torch.save(model.eval().module.state_dict(), os.path.join(save_dir, "final_model.pt"))
    else:
        torch.save(model.eval().state_dict(), os.path.join(save_dir, "final_model.pt"))
    np.save(os.path.join(save_dir, f"losses-{model_name}.npy"), np.array(losses))
    np.save(os.path.join(save_dir, f"steps-{model_name}.npy"), np.array(steps))

    # Plot loss vs. training step graph
    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Loss vs. Training Steps')
    plt.savefig(os.path.join(save_dir, "loss_vs_training_steps.png"), bbox_inches='tight')
    
    return best_val_epoch, best_val_acc, losses[-1]

def train_model_da(model, 
                train_dataloader, 
                val_dataloader, 
                target_dataloader,
                target_val_dataloader,
                scale_factor,
                optimizer, 
                model_name, 
                scheduler = None, 
                epochs=100, 
                device='cuda', 
                save_dir='checkpoints', 
                early_stopping_patience=10, 
                report_interval=5,
                dynamic_weighting = False
            ):
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model.to(device)
    else:
        model.to(device)
    
    warmup = config['parameters']['warmup']
    print("Model Loaded to Device!")
    best_val_acc, best_classification_loss, best_domain_loss, best_total_val_loss = 0, float('inf'), float('inf'), float('inf')
    no_improvement_count = 0
    losses, steps = [], []
    train_classification_losses, train_domain_losses = [], []
    val_losses, val_classification_losses, val_domain_losses = [], [], []
    max_distances, epoch_max_distances = [], []
    js_distances, epoch_js_distances = [], []
    blur_vals, epoch_blur_vals = [], []
    
    print("Training Started!")
    
    if dynamic_weighting:
        sigma_1 = torch.nn.Parameter(torch.tensor(1.0, device=device))
        sigma_2 = torch.nn.Parameter(torch.tensor(1.0, device=device))

        optimizer.add_param_group({'params': [sigma_1, sigma_2]})
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        classification_losses, domain_losses = [], []

        for i, (batch, target_batch) in tqdm(enumerate(zip(train_dataloader, target_dataloader))):
            source_inputs, source_outputs = batch
            source_inputs, source_outputs = source_inputs.to(device).float(), source_outputs.to(device)

            target_inputs, _ = target_batch
            target_inputs = target_inputs.to(device).float()

            optimizer.zero_grad()

            if epoch < warmup:
                _, model_outputs = model(source_inputs)
                classification_loss = F.cross_entropy(model_outputs, source_outputs)
                loss = classification_loss
                domain_loss = None  # No domain loss during warmup
            else:
                concatenated_inputs = torch.cat((source_inputs, target_inputs), dim=0)
                batch_size = source_inputs.size(0)

                # Pass through the model to get features and outputs
                features, model_outputs = model(concatenated_inputs)
                source_features = features[:batch_size]
                target_features = features[batch_size:]
                source_model_outputs = model_outputs[:batch_size]
                
                classification_loss = F.cross_entropy(source_model_outputs, source_outputs)
                
                pairwise_distances = torch.cdist(source_features, target_features, p=2)
                flattened_distances = pairwise_distances.view(-1)
                max_distance = torch.max(flattened_distances)
                max_distances.append(max_distance.detach().cpu().numpy())
                js_distances.append(jensen_shannon_distance(source_features, target_features).nanmean().item())

                dynamic_blur_val = 0.1 * max_distance.detach().cpu().numpy()
                blur_vals.append(dynamic_blur_val)

                domain_loss = sinkhorn_loss(
                    source_features, 
                    target_features, 
                    blur=max(dynamic_blur_val, 0.01),  # Apply lower bound to blur
                    scaling=config['parameters']['scaling'],
                    reach=None
                )

                if dynamic_weighting:
                    loss = (1 / (2 * sigma_1**2)) * classification_loss + (1 / (2 * sigma_2**2)) * domain_loss + torch.log(torch.abs(sigma_1) * torch.abs(sigma_2))
                else:
                    loss = classification_loss + scale_factor * domain_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            if dynamic_weighting:
                sigma_1.data.clamp_(min=1e-3)
                sigma_2.data.clamp_(min=0.25*sigma_1.data.item())
            optimizer.step()

            train_loss += loss.item()
            classification_losses.append(classification_loss.item())
            if epoch >= warmup:
                domain_losses.append(domain_loss.item())
                
        mean_max_distance = np.mean(max_distances)
        epoch_max_distances.append(mean_max_distance) 
        
        mean_blur_val = np.mean(blur_vals)
        epoch_blur_vals.append(mean_blur_val)
        mean_js_distance = np.nanmean(js_distances)
        epoch_js_distances.append(mean_js_distance)

        train_loss /= len(train_dataloader)
        train_classification_loss = np.mean(classification_losses)
        train_domain_loss = np.mean(domain_losses) if domain_losses else None

        losses.append(train_loss)
        train_classification_losses.append(train_classification_loss)
        train_domain_losses.append(train_domain_loss)
        steps.append(epoch + 1)

        # Dynamic weighting logging only after warmup
        if epoch >= warmup and dynamic_weighting:
            print(f"Epoch: {epoch + 1}, sigma_1: {sigma_1.item():.4f}, sigma_2: {sigma_2.item():.4f}")
            print(f"Epoch: {epoch + 1}, Max Distance: {max_distance:.4f}")

        # Adjust logging based on warmup phase
        if epoch < warmup:
            print(f"Epoch: {epoch + 1}, Train Loss: {train_loss:.4e}")
            print(f"Epoch: {epoch + 1}, Classification Loss: {train_classification_loss:.4e}")
        else:
            print(f"Epoch: {epoch + 1}, Train Loss: {train_loss:.4e}")
            print(f"Epoch: {epoch + 1}, Classification Loss: {train_classification_loss:.4e}, Domain Loss: {train_domain_loss:.4e}")

        if scheduler is not None:
            scheduler.step()

        if (epoch + 1) % report_interval == 0:
            model.eval()
            source_correct, target_correct, source_total, target_total, val_loss = 0, 0, 0, 0, 0.0
            val_classification_loss, val_domain_loss = 0.0, 0.0

            with torch.no_grad():
                for i, (batch, target_batch) in enumerate(zip(val_dataloader, target_val_dataloader)):
                    source_inputs, source_outputs = batch
                    source_inputs, source_outputs = source_inputs.to(device).float(), source_outputs.to(device)
                    target_inputs, target_outputs = target_batch
                    target_inputs, target_outputs = target_inputs.to(device).float(), target_outputs.to(device)

                    if epoch < warmup:
                        _, source_preds = model(source_inputs)
                        classification_loss_ = F.cross_entropy(source_preds, source_outputs)
                        combined_loss = classification_loss_
                        domain_loss_ = 0.0  # Set to zero explicitly during warmup
                        target_preds = None  # No target predictions during warmup

                    else:
                        concatenated_inputs = torch.cat((source_inputs, target_inputs), dim=0)
                        batch_size = source_inputs.size(0)

                        # Pass through the model to get features and outputs
                        features, preds = model(concatenated_inputs)
                        source_features = features[:batch_size]
                        target_features = features[batch_size:]
                        source_preds = preds[:batch_size]
                        target_preds = preds[batch_size:]

                        classification_loss_ = F.cross_entropy(source_preds, source_outputs)
                        
                        pairwise_distances = torch.cdist(source_features, target_features, p=2)
                        flattened_distances = pairwise_distances.view(-1)
                        max_distance = torch.max(flattened_distances)

                        dynamic_blur_val = 0.05 * max_distance.detach().cpu().numpy()
                        domain_loss_ = sinkhorn_loss(source_features, 
                                                     target_features, 
                                                     blur=max(dynamic_blur_val, 0.01),  # Apply lower bound to blur
                                                     scaling=config['parameters']['scaling'], 
                                                     reach=None
                                                )
                        
                        combined_loss = classification_loss_ + domain_loss_

                        # Calculate target predictions only after warmup
                        _, target_predicted = torch.max(target_preds.data, 1)
                        target_total += target_outputs.size(0)
                        target_correct += (target_predicted == target_outputs).sum().item()

                    # Common operations for both phases
                    val_loss += combined_loss.item()
                    val_classification_loss += classification_loss_.item()

                    if epoch >= warmup:
                        val_domain_loss += domain_loss_.item()  # Accumulate domain loss only after warmup

                    # These lines should remain unchanged
                    _, source_predicted = torch.max(source_preds.data, 1)
                    source_total += source_outputs.size(0)
                    source_correct += (source_predicted == source_outputs).sum().item()

            source_val_acc = 100 * source_correct / source_total

            if target_total > 0:
                target_val_acc = 100 * target_correct / target_total
            else:
                target_val_acc = 0.0  # Or skip this calculation entirely during warmup

            val_loss /= len(val_dataloader)
            val_classification_loss /= len(val_dataloader)

            if epoch >= warmup:
                val_domain_loss /= len(val_dataloader)  # Normalize domain loss only if accumulated

            val_losses.append(val_loss)
            val_classification_losses.append(val_classification_loss)
            val_domain_losses.append(val_domain_loss)

            lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']

            # Adjust validation logging based on warmup phase
            if epoch < warmup:
                print(f"Epoch: {epoch + 1}, Total Validation Loss: {val_loss:.4f}, Source Validation Accuracy: {source_val_acc:.2f}%, Learning rate: {lr}")
                print(f"Epoch: {epoch + 1}, Validation Classification Loss: {val_classification_loss:.4e}")
            else:
                print(f"Epoch: {epoch + 1}, Total Validation Loss: {val_loss:.4f}, Source Validation Accuracy: {source_val_acc:.2f}%, Learning rate: {lr}, Target Validation Accuracy: {target_val_acc:.2f}%")
                print(f"Epoch: {epoch + 1}, Validation Classification Loss: {val_classification_loss:.4e}, Validation Domain Loss: {val_domain_loss:.4e}")
                
            if val_loss < best_total_val_loss and epoch >= warmup:
                best_total_val_loss = val_loss
                best_val_epoch = epoch + 1
                if torch.cuda.device_count() > 1:
                    torch.save(model.eval().module.state_dict(), os.path.join(save_dir, "best_model_total_val_loss.pt"))
                else:
                    torch.save(model.eval().state_dict(), os.path.join(save_dir, "best_model_total_val_loss.pt"))
                print(f"Saved best total validation loss model at epoch {best_val_epoch}")
                
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
                print(f"Saved best validation accuracy model at epoch {best_val_acc_epoch}")

            # Check and save the model with lowest classification loss
            if val_classification_loss <= best_classification_loss and epoch >= warmup:
                best_classification_loss = val_classification_loss
                best_classification_loss_epoch = epoch + 1
                model_path = os.path.join(save_dir, "best_model_classification_loss.pt")
                if torch.cuda.device_count() > 1:
                    torch.save(model.eval().module.state_dict(), model_path)
                else:
                    torch.save(model.eval().state_dict(), model_path)
                print(f"Saved lowest classification loss model at epoch {best_classification_loss_epoch}")

            # Check and save the model with lowest domain loss
            if val_domain_loss <= best_domain_loss and epoch >= warmup:
                best_domain_loss = val_domain_loss
                best_domain_epoch = epoch + 1
                model_path = os.path.join(save_dir, "best_model_domain_loss.pt")
                if torch.cuda.device_count() > 1:
                    torch.save(model.eval().module.state_dict(), model_path)
                else:
                    torch.save(model.eval().state_dict(), model_path)
                print(f"Saved lowest domain loss model at epoch {best_domain_epoch}")

            if no_improvement_count >= early_stopping_patience:
                print(f"Early stopping after {early_stopping_patience} epochs without improvement in accuracy.")
                break
    
    # Save final model
    if torch.cuda.device_count() > 1:
        torch.save(model.eval().module.state_dict(), os.path.join(save_dir, "final_model.pt"))
    else:
        torch.save(model.eval().state_dict(), os.path.join(save_dir, "final_model.pt"))

    loss_dir = os.path.join(save_dir, 'losses')
    if not os.path.exists(loss_dir):
        os.makedirs(loss_dir)
        
    np.save(os.path.join(loss_dir, f"losses-{model_name}.npy"), np.array(losses))
    np.save(os.path.join(loss_dir, f"train_classification_losses-{model_name}.npy"), np.array(train_classification_losses))
    np.save(os.path.join(loss_dir, f"train_domain_losses-{model_name}.npy"), np.array(train_domain_losses))
    np.save(os.path.join(loss_dir, f"val_losses-{model_name}.npy"), np.array(val_losses))
    np.save(os.path.join(loss_dir, f"val_classification_losses-{model_name}.npy"), np.array(val_classification_losses))
    np.save(os.path.join(loss_dir, f"val_domain_losses-{model_name}.npy"), np.array(val_domain_losses))
    np.save(os.path.join(loss_dir, f"steps-{model_name}.npy"), np.array(steps))
    np.save(os.path.join(loss_dir, f"max_distances-{model_name}.npy"), np.array(max_distances))
    np.save(os.path.join(loss_dir, f"blur_vals-{model_name}.npy"), np.array(blur_vals))
    np.save(os.path.join(loss_dir, f"js_distances-{model_name}.npy"), np.array(js_distances))
    np.save(os.path.join(loss_dir, f"epoch_max_distances-{model_name}.npy"), np.array(epoch_max_distances))
    np.save(os.path.join(loss_dir, f"epoch_blur_vals-{model_name}.npy"), np.array(epoch_blur_vals))
    np.save(os.path.join(loss_dir, f"epoch_js_distances-{model_name}.npy"), np.array(epoch_js_distances))
    
    
    # Plotting the losses
    plt.figure(figsize=(14, 8))
    
    steps = np.array(steps)
    validation_steps = steps[::report_interval]
    losses = np.array(losses)
    train_classification_losses = np.array(train_classification_losses)
    train_domain_losses = np.array(train_domain_losses)
    val_losses = np.array(val_losses)
    val_classification_losses = np.array(val_classification_losses)
    val_domain_losses = np.array(val_domain_losses)
    
    # Plot Training Losses
    plt.subplot(2, 1, 1)
    plt.plot(steps, losses, label='Train Total Loss')
    plt.plot(steps, train_classification_losses, label='Train Classification Loss')
    plt.plot(steps, train_domain_losses, label='Train Domain Loss')
    plt.axvline(x=best_val_epoch, color='b', linestyle='--', label='Best Val Epoch')
    plt.axvline(x=best_classification_loss_epoch, color='y', linestyle='--', label='Best Classification Epoch')
    plt.axvline(x=best_domain_epoch, color='g', linestyle='--', label='Best Domain Epoch')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.yscale('log')
    plt.legend()

    # Plot Validation Losses
    plt.subplot(2, 1, 2)
    plt.plot(validation_steps, val_losses, label='Validation Total Loss')
    plt.plot(validation_steps, val_classification_losses, label='Validation Classification Loss')
    plt.plot(validation_steps, val_domain_losses, label='Validation Domain Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Validation Losses')
    plt.yscale('log')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(loss_dir, f"losses_plot-{model_name}.png"))
    plt.close()
    
    plt.figure(figsize=(10, 5))
    
    plt.plot(steps, epoch_max_distances)
    plt.axvline(x=best_val_epoch, color='b', linestyle='--', label='Best Val Epoch')
    plt.axvline(x=best_classification_loss_epoch, color='y', linestyle='--', label='Best Classification Epoch')
    plt.axvline(x=best_domain_epoch, color='g', linestyle='--', label='Best Domain Epoch')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Max Distance')
    plt.title('Max Distance vs. Training Steps')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(loss_dir, f"max_distance_plot-{model_name}.png"))
    plt.close()
    
    plt.figure(figsize=(10, 5))
    
    plt.plot(steps, epoch_blur_vals)
    plt.axhline(y=0.01, color='r', linestyle='--')
    plt.axhline(y=0.05, color='g', linestyle='--')
    plt.axvline(x=best_val_epoch, color='b', linestyle='--', label='Best Val Epoch')
    plt.axvline(x=best_classification_loss_epoch, color='y', linestyle='--', label='Best Classification Epoch')
    plt.axvline(x=best_domain_epoch, color='g', linestyle='--', label='Best Domain Epoch')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Blur Value')
    plt.title('Blur Value vs. Training Steps')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(loss_dir, f"blur_value_plot-{model_name}.png"))
    plt.close()
    
    plt.figure(figsize=(10, 5))
    
    plt.plot(steps, epoch_js_distances)
    plt.axvline(x=best_val_epoch, color='b', linestyle='--', label='Best Val Epoch')
    plt.axvline(x=best_classification_loss_epoch, color='y', linestyle='--', label='Best Classification Epoch')
    plt.axvline(x=best_domain_epoch, color='g', linestyle='--', label='Best Domain Epoch')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('JS Distance')
    plt.title('JS Distance vs. Training Steps')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(loss_dir, f"js_distance_plot-{model_name}.png"))
    plt.close()

    return best_val_epoch, best_val_acc, best_classification_loss_epoch, best_classification_loss, best_domain_epoch, best_domain_loss, losses[-1]

def main(config):
    num_classes = config['num_classes']
    # model = d4_model(num_classes) if config['model'] == 'D4' else cnn(num_classes)
    # model = d4_mnistm(num_classes) if config['model'] == 'D4' else cnn_mnistm(num_classes)
    model_name = str(config['model'])
    model = mmnistm_models[model_name](num_classes=num_classes)
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params_to_optimize, 
                            lr = config['parameters']['lr'], 
                            weight_decay = config['parameters']['weight_decay']
                        )

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                                               milestones = config['parameters']['milestones'],
                                               gamma=config['parameters']['lr_decay']
                                            )
        
    # Define transformations (for blobs and shapes dataset)
    # train_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.RandomRotation(180),
    #     transforms.Resize(100),
    #     transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    #     transforms.RandomHorizontalFlip(p=0.3),
    #     transforms.RandomVerticalFlip(p=0.3),
    #     transforms.Normalize(mean=(0.5, ), std=(0.5,))
    # ])

    # val_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=(0.5, ), std=(0.5,)),
    #     transforms.Resize(100)
    # ])
    
    ## define transformations for MNIST dataset
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(180),
        transforms.Resize(32),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        transforms.Resize(32)
    ])

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
    train_dataset = MnistM(input_path=config['train_data']['input_path'], 
                        output_path=config['train_data']['output_path'], 
                        transform=train_transform)

    # Split source dataset into train and validation sets
    train_dataset, val_dataset = split_dataset(train_dataset, 
                                            val_size=config['parameters']['val_size'],
                                            train_transform=train_transform, 
                                            val_transform=val_transform)

    if config['DA']:
        # Load target dataset
        target_dataset = MnistM(input_path=config['train_data']['target_input_path'], 
                                output_path=config['train_data']['target_output_path'], 
                                transform=train_transform)
        
        # Split target dataset into train and validation sets
        target_dataset, val_target_dataset = split_dataset(target_dataset, 
                                                        val_size=config['parameters']['val_size'], 
                                                        train_transform=train_transform, 
                                                        val_transform=val_transform)

    end = time.time()
    print(f"Datasets loaded and split in {end - start} seconds")

    # Dataloaders can be created if needed
    train_dataloader = DataLoader(train_dataset, batch_size=config['parameters']['batch_size'], shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['parameters']['batch_size'], shuffle=False, pin_memory=True)
    if config['DA']:
        target_dataloader = DataLoader(target_dataset, batch_size=config['parameters']['batch_size'], shuffle=True, pin_memory=True)
        target_val_dataloader = DataLoader(val_target_dataset, batch_size=config['parameters']['batch_size'], shuffle=False, pin_memory=True)

    timestr = time.strftime("%Y%m%d-%H%M%S")
    
    if config['DA']:
        save_dir = config['save_dir'] + config['model'] + '_DA_' + timestr
        best_val_epoch, best_val_acc, best_classification_epoch, best_classification_loss, best_domain_epoch, best_domain_loss, final_loss = train_model_da(model=model,
                                                                                                                                                    train_dataloader=train_dataloader, 
                                                                                                                                                    val_dataloader=val_dataloader,
                                                                                                                                                    target_dataloader=target_dataloader,
                                                                                                                                                    target_val_dataloader=target_val_dataloader,
                                                                                                                                                    scale_factor=config['parameters']['scale_factor'],
                                                                                                                                                    optimizer=optimizer, 
                                                                                                                                                    model_name=model_name, 
                                                                                                                                                    scheduler=scheduler, 
                                                                                                                                                    epochs=config['parameters']['epochs'], 
                                                                                                                                                    device=device, 
                                                                                                                                                    save_dir=save_dir,
                                                                                                                                                    early_stopping_patience=config['parameters']['early_stopping'], 
                                                                                                                                                    report_interval=config['parameters']['report_interval'],
                                                                                                                                                    dynamic_weighting=config['dynamic_weighting']
                                                                                   )
        print('Training Done')
        config['best_val_acc'] = best_val_acc
        config['best_val_epoch'] = best_val_epoch
        config['final_loss'] = float(final_loss)
        # config['feature_fields'] = feature_fields
        config['best_classification_epoch'] = best_classification_epoch
        config['best_classification_loss'] = best_classification_loss
        config['best_domain_epoch'] = best_domain_epoch
        config['best_domain_loss'] = best_domain_loss
        
    else:
        save_dir = config['save_dir'] + config['model'] + '_' + timestr
        best_val_epoch, best_val_acc, final_loss = train_model(model=model, 
                                                           train_dataloader=train_dataloader, 
                                                           val_dataloader=val_dataloader,
                                                           optimizer=optimizer, 
                                                           model_name=model_name, 
                                                           scheduler=scheduler, 
                                                           epochs=config['parameters']['epochs'], 
                                                           device=device, 
                                                           save_dir=save_dir,
                                                           early_stopping_patience=config['parameters']['early_stopping'], 
                                                           report_interval=config['parameters']['report_interval'],
                                                    
                                                        )
        print('Training Done')
        config['best_val_acc'] = best_val_acc
        config['best_val_epoch'] = best_val_epoch
        config['final_loss'] = final_loss
        # config['feature_fields'] = feature_fields

    file = open(f'{save_dir}/config.yaml',"w")
    yaml.dump(config, file)
    file.close()
    
if __name__ == '__main__':

    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser(description = 'Train the models')
    parser.add_argument('--config', metavar = 'config', required=True,
                    help='Location of the config file')

    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    set_all_seeds(config['seed'])

    main(config)

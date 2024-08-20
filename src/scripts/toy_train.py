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
from toy_model_simple import d4_model, feature_fields, cnn
from toy_dataset import Shapes
from tqdm import tqdm
import random
import geomloss

def sinkhorn_loss(x, 
                  y,
            ):
    loss = geomloss.SamplesLoss(loss='sinkhorn', scaling=0.8)
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
    best_val_acc, best_classification_loss, best_domain_loss = 0, float('inf'), float('inf')
    no_improvement_count = 0
    losses, steps = [], []
    train_classification_losses, train_domain_losses = [], []
    val_losses, val_classification_losses, val_domain_losses = [], [], []
    
    print("Training Started!")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        classification_losses, domain_losses = [], []
        
        for i, (batch, target_batch) in tqdm(enumerate(zip(train_dataloader, target_dataloader))):
            inputs, targets = batch
            inputs, targets = inputs.to(device).float(), targets.to(device)
            
            target_inputs, _ = target_batch
            target_inputs = target_inputs.to(device).float()
            
            features, outputs = model(inputs)
            target_features, _ = model(target_inputs)
                    
            classificaiton_loss = F.cross_entropy(outputs, targets)
            domain_loss = sinkhorn_loss(features, target_features)            
            loss = classificaiton_loss + scale_factor * domain_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            classification_losses.append(classificaiton_loss.item())
            domain_losses.append(domain_loss.item())

        train_loss /= len(train_dataloader)
        train_classification_loss = np.mean(classification_losses)
        train_domain_loss = np.mean(domain_losses)
        
        losses.append(train_loss)
        train_classification_losses.append(train_classification_loss)
        train_domain_losses.append(train_domain_loss)
        steps.append(epoch + 1)
        
        print(f"Epoch: {epoch + 1}, Train Loss: {train_loss:.4e}")
        print(f"Epoch: {epoch + 1}, Classification Loss: {train_classification_loss:.4e}, Domain Loss: {train_domain_loss:.4e}")

        if scheduler is not None:
            scheduler.step()

        if (epoch + 1) % report_interval == 0:
            model.eval()
            correct, total, val_loss = 0, 0, 0.0
            val_classification_loss, val_domain_loss = 0.0, 0.0

            with torch.no_grad():
                for i, (batch, target_batch) in enumerate(zip(val_dataloader, target_val_dataloader)):
                    inputs, targets = batch
                    inputs, targets = inputs.to(device).float(), targets.to(device)
                    target_inputs, _ = target_batch
                    target_inputs = target_inputs.to(device).float()
                    features, outputs = model(inputs)
                    target_features, _ = model(target_inputs)
                    features = features.view(features.size(0), -1)
                    target_features = target_features.view(target_features.size(0), -1)
                    
                    classification_loss_ = F.cross_entropy(outputs, targets)
                    domain_loss_ = sinkhorn_loss(features, target_features)
                    combined_loss = classification_loss_ + scale_factor * domain_loss_
                    
                    val_loss += combined_loss.item()
                    val_classification_loss += classification_loss_.item()
                    val_domain_loss += domain_loss_.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()

            val_acc = 100 * correct / total
            val_loss /= len(val_dataloader)
            val_classification_loss /= len(val_dataloader)
            val_domain_loss /= len(val_dataloader)
            val_losses.append(val_loss)
            val_classification_losses.append(val_classification_loss)
            val_domain_losses.append(val_domain_loss)
            
            lr = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]['lr']
            print(f"Epoch: {epoch + 1}, Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%, Learning rate: {lr}")
            print(f"Epoch: {epoch + 1}, Classification Loss: {val_classification_loss:.4e}, Domain Loss: {val_domain_loss:.4e}")

            # Check and save the model with best validation accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_epoch = epoch + 1
                model_path = os.path.join(save_dir, "best_model_val_acc.pt")
                if torch.cuda.device_count() > 1:
                    torch.save(model.eval().module.state_dict(), model_path)
                else:
                    torch.save(model.eval().state_dict(), model_path)
                print(f"Saved best validation accuracy model at epoch {best_val_epoch}")

            # Check and save the model with lowest classification loss
            if val_classification_loss < best_classification_loss:
                best_classification_loss = val_classification_loss
                best_classification_epoch = epoch + 1
                model_path = os.path.join(save_dir, "best_model_classification_loss.pt")
                if torch.cuda.device_count() > 1:
                    torch.save(model.eval().module.state_dict(), model_path)
                else:
                    torch.save(model.eval().state_dict(), model_path)
                print(f"Saved lowest classification loss model at epoch {best_classification_epoch}")

            # Check and save the model with lowest domain loss
            if val_domain_loss < best_domain_loss:
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
        # Saving losses and steps
    np.save(os.path.join(loss_dir, f"losses-{model_name}.npy"), np.array(losses))
    np.save(os.path.join(loss_dir, f"train_classification_losses-{model_name}.npy"), np.array(train_classification_losses))
    np.save(os.path.join(loss_dir, f"train_domain_losses-{model_name}.npy"), np.array(train_domain_losses))
    np.save(os.path.join(loss_dir, f"val_losses-{model_name}.npy"), np.array(val_losses))
    np.save(os.path.join(loss_dir, f"val_classification_losses-{model_name}.npy"), np.array(val_classification_losses))
    np.save(os.path.join(loss_dir, f"val_domain_losses-{model_name}.npy"), np.array(val_domain_losses))
    np.save(os.path.join(loss_dir, f"steps-{model_name}.npy"), np.array(steps))
    
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
    plt.show()

    return best_val_epoch, best_val_acc, best_classification_epoch, best_classification_loss, best_domain_epoch, best_domain_loss, losses[-1]

def main(config):
    model = cnn()
    model_name = 'D4'
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params_to_optimize, 
                            lr = config['parameters']['lr'], 
                            weight_decay = config['parameters']['weight_decay']
                        )

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
                                               milestones = config['parameters']['milestones'],
                                               gamma=config['parameters']['lr_decay']
                                            )
        
    # Define transformations
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.RandomRotation(180),
        transforms.Resize(100),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.Normalize(mean=(0.5, ), std=(0.5,))
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, ), std=(0.5,)),
        transforms.Resize(100)
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
    train_dataset = Shapes(input_path=config['train_data']['input_path'], 
                        output_path=config['train_data']['output_path'], 
                        transform=train_transform)

    # Split source dataset into train and validation sets
    train_dataset, val_dataset = split_dataset(train_dataset, 
                                            val_size=config['parameters']['val_size'],
                                            train_transform=train_transform, 
                                            val_transform=val_transform)

    if config['DA']:
        # Load target dataset
        target_dataset = Shapes(input_path=config['train_data']['target_input_path'], 
                                output_path=config['train_data']['target_output_path'], 
                                transform=val_transform)
        
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
                                                                                                                                                    report_interval=config['parameters']['report_interval']
                                                                                   )
        print('Training Done')
        config['best_val_acc'] = best_val_acc
        config['best_val_epoch'] = best_val_epoch
        config['final_loss'] = float(final_loss)
        config['feature_fields'] = feature_fields
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
        config['feature_fields'] = feature_fields

    file = open(f'{save_dir}/config.yaml',"w")
    yaml.dump(config, file)
    file.close()
    
if __name__ == '__main__':

    device = ('cuda' if torch.cuda.is_available() else 'cpu')
        
    set_all_seeds(42)

    parser = argparse.ArgumentParser(description = 'Train the models')
    parser.add_argument('--config', metavar = 'config', required=True,
                    help='Location of the config file')

    args = parser.parse_args()


    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    main(config)

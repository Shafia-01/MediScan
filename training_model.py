import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import WeightedRandomSampler
from torchvision import datasets, transforms, models


if __name__ == '__main__':
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'data'
    image_datasets = {
        split: datasets.ImageFolder(os.path.join(data_dir, split), data_transforms[split])
        for split in ['train', 'val']
    }

    train_targets = np.array(image_datasets['train'].targets)
    class_counts = np.bincount(train_targets)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[train_targets]
    sampler = WeightedRandomSampler(weights=torch.from_numpy(sample_weights).double(),
                                    num_samples=len(sample_weights),
                                    replacement=True)

    use_cuda = torch.cuda.is_available()
    num_workers = 0
    pin_memory = True if use_cuda else False

    dataloaders = {
        'train': torch.utils.data.DataLoader(
            image_datasets['train'], batch_size=16, sampler=sampler, num_workers=num_workers, pin_memory=pin_memory
        ),
        'val': torch.utils.data.DataLoader(
            image_datasets['val'], batch_size=16, shuffle=False, num_workers=num_workers, pin_memory=pin_memory
        ),
    }

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    print({"sizes": dataset_sizes, "classes": class_names})

    num_classes = len(class_names)
    device = torch.device('cuda' if use_cuda else 'cpu')

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    model = model.to(device)

    for name, param in model.named_parameters():
        param.requires_grad = name.startswith('fc')

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    best_val_loss = float('inf')
    best_state_dict = None
    patience = 4
    epochs_no_improve = 0

    total_epochs = 12
    warmup_epochs = 3

    for epoch in range(total_epochs):
        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f"Epoch {epoch+1}/{total_epochs} | {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        scheduler.step()

        if epoch + 1 == warmup_epochs:
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
            print("Unfroze all layers for fine-tuning.")

        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for inputs, labels in dataloaders['val']:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
        val_loss = running_loss / dataset_sizes['val']

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            print(f"New best val loss: {best_val_loss:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    torch.save(model.state_dict(), 'image_classification_model.pth')
    class_to_idx = image_datasets['train'].class_to_idx
    with open('class_to_idx.json', 'w') as f:
        json.dump(class_to_idx, f)
    print("Saved best model weights and class_to_idx.json")
    
    
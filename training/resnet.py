import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

from torch.optim.lr_scheduler import StepLR

import numpy as np
import matplotlib.pyplot as plt
import random
import os
import shutil

from src.vgg import VGG
from src.train import train
from src.eval import eval

# Modification to use ResNet
def main(model: str = 'ResNet18', 
         classes: int = 4, 
         image_size: int = 100, 
         pretrained: bool = True) -> None:

    # Fix random seed
    random_seed = 9999
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
    else:
        device = torch.device('cpu')

    # Image preprocessing
    train_transform = transforms.Compose([
        transforms.Resize((image_size + 20, image_size + 20)),  # Resize the original image slightly larger
        transforms.RandomCrop((image_size, image_size)),       # Random cropping
        transforms.RandomHorizontalFlip(p=0.5),               # Horizontal flip (50% probability)
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Translation
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5), # Perspective transformation
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalization
    ])

    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Simple resizing for test data
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalization
    ])

    # Load dataset
    dataset = datasets.ImageFolder(root='./train', transform=test_transform)
    test_dataset = datasets.ImageFolder(root='./test', transform=test_transform)

    # Manually set class labels
    custom_class_to_idx = {'barefoot': 0, 'others': 1, 'shoes': 2, 'socks': 3}  # Modify as needed
    dataset.class_to_idx = custom_class_to_idx
    test_dataset.class_to_idx = custom_class_to_idx

    # Create a mapping from index to class name
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    print("Index-to-class mapping:", idx_to_class)

    # Split dataset
    train_len = int(len(dataset) * 0.9)
    val_len = len(dataset) - train_len
    train_dataset, val_dataset = random_split(dataset, [train_len, val_len])
    train_dataset.dataset.transform = train_transform

    # Prepare data loaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=15, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=15, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=15, pin_memory=True)

    # Create model (using ResNet18)
    if model == 'ResNet18':
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, classes)  # Modify the final fully connected layer

    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

    # Training
    train_result = []
    eval_result = []
    for epoch in range(1, 51):  # Number of epochs
        train_loss, train_acc = train(epoch, model, optimizer, criterion, train_loader, device)
        eval_loss, eval_acc = eval(epoch, model, criterion, val_loader, device)
        train_result.append((train_acc, train_loss))
        eval_result.append((eval_acc, eval_loss))

        scheduler.step()

    # Save results
    train_acc, train_loss = zip(*train_result)
    eval_acc, eval_loss = zip(*eval_result)
    plt.figure()
    plt.plot(train_acc, label='train_acc')
    plt.plot(eval_acc, label='eval_acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('acc.png')

    plt.figure()
    plt.plot(train_loss, label='train_loss')
    plt.plot(eval_loss, label='eval_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss.png')

    # Testing
    model.eval()
    class_correct = [0 for _ in range(classes)]
    class_total = [0 for _ in range(classes)]

    # Create folders for misclassified images
    for cls in idx_to_class.values():
        os.makedirs(f'./misclassified/{cls}', exist_ok=True)

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            batch_paths = [path for path, _ in test_dataset.samples[i * test_loader.batch_size:(i+1) * test_loader.batch_size]]
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            for j in range(len(labels)):
                label = labels[j].item()
                pred = predicted[j].item()
                if label == pred:
                    class_correct[label] += 1
                else:
                    src_path = batch_paths[j]
                    dst_path = f'./misclassified/{idx_to_class[pred]}/{os.path.basename(src_path)}'
                    shutil.copy(src_path, dst_path)
                    print(f'[Misclassified] Correct: {idx_to_class[label]} -> Predicted: {idx_to_class[pred]} | Image: {src_path}')
                class_total[label] += 1

    for i in range(classes):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(f'Accuracy for {idx_to_class[i]}: {acc:.2f}%')
        else:
            print(f'No samples for {idx_to_class[i]}.')

    total_correct = sum(class_correct)
    total_images = sum(class_total)
    if total_images > 0:
        overall_acc = 100 * total_correct / total_images
        print(f'\nOverall Accuracy: {overall_acc:.2f}%')
    else:
        print('No images in the test set.')

    torch.save(model.state_dict(), '../ros2_ws/final_weight.pth')
    print(f'Model saved to ./final_weight_resnet.pth')

if __name__ == '__main__':
    main()

import torch
from torch import nn
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
import time
import numpy as np
import random
import os
import shutil

def test(model: nn.Module, 
         criterion: nn.Module, 
         loader: DataLoader, 
         device: torch.device, 
         idx_to_class: dict) -> None:
    """
    Test the model's performance on a dataset.

    Parameters:
    - model (torch.nn.Module): The model to be tested.
    - criterion (torch.nn.Module): The loss function used for testing.
    - loader (torch.utils.data.DataLoader): DataLoader for the dataset to be tested on.
    - device (torch.device): The device to which tensors should be moved before computation.
    - idx_to_class (dict): Mapping from class indices to class names.
    """
    
    model.eval()

    test_loss = 0  # Accumulated test loss
    correct = 0    # Count of correctly predicted samples
    total = 0      # Total samples processed
    class_correct = [0 for _ in range(len(idx_to_class))]
    class_total = [0 for _ in range(len(idx_to_class))]

    with torch.no_grad():
        print("Start testing...")
        start = time.perf_counter()
        for _, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            for i in range(len(targets)):
                label = targets[i].item()
                pred = predicted[i].item()
                class_total[label] += 1
                if label == pred:
                    class_correct[label] += 1

        end = time.perf_counter()
        
    print(f'Test_Loss: {test_loss/len(loader):.4f} | Test_Accuracy: {100.*correct/total:.2f}% | Elapsed Time: {end - start:.2f} seconds')

    for i in range(len(idx_to_class)):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            print(f'{idx_to_class[i]} accuracy: {class_acc:.2f}%')
        else:
            print(f'{idx_to_class[i]} has no samples in the test set.')

    overall_acc = 100 * correct / total
    print(f'Overall accuracy: {overall_acc:.2f}%')

# Only test
if __name__ == '__main__':
    image_size = 100  # This should match the image size used during training

    # Set random seed for reproducibility
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

    print(f"Used device: {device}")

    # Preprocessing for test data
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Prepare test dataset and loader
    test_dataset = datasets.ImageFolder(root='./test', transform=test_transform)

    # Manually set class_to_idx for consistency with the training phase
    custom_class_to_idx = {'barefoot': 0, 'others': 1, 'shoes': 2, 'socks': 3}
    test_dataset.class_to_idx = custom_class_to_idx

    idx_to_class = {v: k for k, v in test_dataset.class_to_idx.items()}
    print("Index-to-class mapping:", idx_to_class)

    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=15, pin_memory=True)

    # Initialize the model (ResNet18)
    model = models.resnet18(pretrained=True)  # ResNet18
    model.fc = nn.Linear(model.fc.in_features, len(custom_class_to_idx))  # Modify the last layer for the number of classes
    model = model.to(device)

    # Load trained model weights
    model.load_state_dict(torch.load('./final_weight.pth'))

    # Define the loss function
    criterion = nn.CrossEntropyLoss().to(device)

    # Test the model
    test(model, criterion, test_loader, device, idx_to_class)


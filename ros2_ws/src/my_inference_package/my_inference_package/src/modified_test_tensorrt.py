import torch
from torch import nn
import torch_tensorrt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import time
import numpy as np
import random
import os

from torchvision import models

def test(model: nn.Module, 
         criterion: nn.Module, 
         loader: DataLoader, 
         device: torch.device) -> None:
    """
    Test the model's performance on a dataset.

    Parameters:
    - model (torch.nn.Module): The model to be tested.
    - criterion (torch.nn.Module): The loss function used for testing.
    - loader (torch.utils.data.DataLoader): DataLoader for the dataset to be tested on.
    - device (torch.device): The device to which tensors should be moved before computation.
    """
    
    # Set the model to evaluation mode. In this mode, operations like dropout are disabled.
    model.eval()

    test_loss = 0  # Accumulated test loss
    correct = 0    # Count of correctly predicted samples
    total = 0      # Total samples processed

    # Disable gradient computations, as we are in evaluation mode and don't need gradients
    print("start_test")
    with torch.no_grad():
        start = time.perf_counter()
        for _, (inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        end = time.perf_counter()

    print(f'Test_Loss: {test_loss/len(loader)} | Test_Accuracy: {100.*correct/total} | Elapsed Time: {end - start}')

# Only test
if __name__ == '__main__':
    image_size = 100  # 訓練時と一致させる

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

    print(f"used_device: {device}")
    
    # Preprocessing for test data
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # 訓練時と一致させる
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Prepare test data loader (訓練時と同じデータセットを使用)
    test_dataset = datasets.ImageFolder(root='./test', transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    # Initialize and load model weights
    model = models.resnet18(pretrained=False)  # pretrained=Falseで訓練済みモデルを使用
    model.fc = nn.Linear(model.fc.in_features, 4)  # 訓練時に設定したクラス数（4クラス）
    model.load_state_dict(torch.load('./final_weight.pth'))  # 訓練済みモデルの重みを読み込む
    model = model.float().eval().to(device)  # モデルをfloatに変換し、評価モードに設定

    # Define the loss function
    criterion = nn.CrossEntropyLoss().to(device)

    ##--torch_tensorrt--##
    # For TensorRT optimization
    inputs = [torch.randn((1, 3, image_size, image_size)).to("cuda")]
    enabled_precisions = {torch.float}  # TensorRTの精度を設定
    debug = False  # デバッグログを無効に

    # Trace and optimize the model with TensorRT
    traced_script_module = torch.jit.trace(model, inputs)
    optimized_model = torch_tensorrt.compile(
        traced_script_module,
        inputs=inputs,
        enabled_precisions=enabled_precisions,
        debug=debug,
    )
    ##--torch_tensorrt--##
    
    # Test the optimized model
    test(optimized_model, criterion, test_loader, device)


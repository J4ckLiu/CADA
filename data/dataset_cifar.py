import os
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import transforms, datasets




def build_loader(data_dir, model, device, train=False, train_num=2000, cal_num=4000, batch_size=256, num_workers=8):
    """
    Build DataLoaders from a dataset, perform inference on the specified device, and return new DataLoaders with logits and labels on CPU.
    
    Args:
        data_dir (str): Path to the dataset directory.
        model (nn.Module): PyTorch model for inference.
        device (str): Device for inference ('cuda' or 'cpu').
        train (bool): If True, return only trainloader; else return all DataLoaders.
        train_num (int): Number of samples for train set.
        cal_num (int): Number of samples for calibration set.
        batch_size (int): Batch size for DataLoaders.
        num_workers (int): Number of workers for DataLoaders.
    
    Returns:
        tuple: (trainloader, calibloader, testloader) - New DataLoaders with logits and labels on CPU.
    """
    # Define transform for ImageNet-style data
    transform_imagenet_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    
    # Build original DataLoaders
    testset = datasets.CIFAR100(root=data_dir, train=False, transform = transform_imagenet_test, download=True)
    dataset_length = len(testset)
    
    # Validate dataset size
    if train_num + cal_num > dataset_length:
        raise ValueError(f"train_num ({train_num}) + cal_num ({cal_num}) exceeds dataset size ({dataset_length})")
    
    # Split dataset
    trainset, testset = random_split(testset, [train_num, dataset_length - train_num])
    dataset_length = len(testset)
    calibset, testset = random_split(testset, [cal_num, dataset_length - cal_num])
    
    # Create original DataLoaders
    trainloader = DataLoader(dataset=trainset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    calibloader = DataLoader(dataset=calibset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    testloader = DataLoader(dataset=testset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    
    # Ensure model is on the specified device
    model = model.to(device)
    model.eval()  # Set model to evaluation mode
    
    train_logits, train_labels = [], []
    calib_logits, calib_labels = [], []
    test_logits, test_labels = [], []
    
    def process_loader(loader, logit_list, label_list):
        with torch.no_grad():
            for data, labels in loader:
                data = data.to(device)  
                outputs = model(data)  
                logit_list.append(outputs.cpu())  
                label_list.append(labels.cpu())  
    
    print("Processing trainloader...")
    process_loader(trainloader, train_logits, train_labels)
    train_logits = torch.cat(train_logits, dim=0)
    train_labels = torch.cat(train_labels, dim=0)
    train_dataset = TensorDataset(train_logits, train_labels)  
    new_trainloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,  
        num_workers=num_workers
    )

    if train:
        return new_trainloader
    
    print("Processing calibloader...")
    process_loader(calibloader, calib_logits, calib_labels)
    print("Processing testloader...")
    process_loader(testloader, test_logits, test_labels)
    
    # Concatenate logits and labels
    calib_logits = torch.cat(calib_logits, dim=0)
    calib_labels = torch.cat(calib_labels, dim=0)
    test_logits = torch.cat(test_logits, dim=0)
    test_labels = torch.cat(test_labels, dim=0)
    
    # Create new TensorDatasets
    calib_dataset = TensorDataset(calib_logits, calib_labels)  # Tensors are on CPU
    test_dataset = TensorDataset(test_logits, test_labels)  # Tensors are on CPU
    
    # Create new DataLoaders with same parameters
    new_calibloader = DataLoader(
        dataset=calib_dataset,
        batch_size=batch_size,
        shuffle=False,  # Match original calibloader
        num_workers=num_workers
    )
    new_testloader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,  # Match original testloader
        num_workers=num_workers
    )
    
    return new_trainloader, new_calibloader, new_testloader



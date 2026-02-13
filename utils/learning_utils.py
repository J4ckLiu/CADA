import os
import re
from predictor.SplitPredictor import Predictor
import torch
from tqdm import tqdm
from itertools import cycle


def validate_and_select(model, val_loader, model_name, conformal, alpha, dataset,save_dir, class_num, device):
    pattern = re.compile(rf'^{model_name}_{dataset}_(\d+)iter\.pth$')
    files = []
    for file in os.listdir(save_dir):
        match = pattern.match(file)
        if match:
            iteration_count = int(match.group(1))  
            files.append((iteration_count, os.path.join(save_dir, file)))
    files.sort(key=lambda x: x[0])  
    size_optimal = float('inf')
    iter_optimal = -1
    if len(files)==0:
        print(f"Adapter has not been tuned for {model_name} on {dataset}")
        exit()
        
    print(f"Model Selection for {conformal} at error rate equals {alpha}")
    for iteration_count, file_path in files:
        model.base_model.load_state_dict(torch.load(file_path))
        predictor = Predictor(model, conformal, alpha, device)
        predictor.calibrate(val_loader)
        result  = predictor.evaluate(val_loader,alpha,class_num)
        record = result['Size']
        print(record)
        print(f"iters: {iteration_count} finished")
        if record<size_optimal:
            size_optimal = record
            iter_optimal = iteration_count

    print(f"Model Selection Over, optimal iter: {iter_optimal}")
    return iter_optimal





def train_continuous(model, train_loader, criterion, optimizer, model_name, dataset, max_iter, save_iters, savedir, device):
    running_loss = 0.0
    total = 0
    iteration_count = 0

    # Create an infinite iterator over the data loader
    data_iterator = cycle(train_loader)

    # Progress bar for max_iter
    with tqdm(total=max_iter) as progress_bar:
        while iteration_count < max_iter:
            # Get the next batch
            inputs, labels = next(data_iterator)
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Update running loss and total
            running_loss += loss.item()
            total += labels.size(0)

            # Save model at specified iterations
            if iteration_count in save_iters and iteration_count > 0:
                save_path = os.path.join(savedir, f'{model_name}_{dataset}_{iteration_count}iter.pth')
                torch.save(model.base_model.state_dict(), save_path)

            # Update progress bar with average loss
            avg_loss = running_loss / (iteration_count + 1)
            progress_bar.set_description(f'Average Loss (over iteration): {avg_loss:.6f}')
            progress_bar.update(1)

            # Increment iteration count
            iteration_count += 1

    return running_loss / iteration_count
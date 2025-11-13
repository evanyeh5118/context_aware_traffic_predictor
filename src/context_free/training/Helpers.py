import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ..config import TrainingConfig

def trainModelHelper(
        model, criterion, optimizer, train_loader, test_loader, trainingConfig: TrainingConfig, 
        verbose=False, model_path=None):
    num_epochs = trainingConfig.num_epochs
    teacher_forcing_ratio = trainingConfig.teacher_forcing_ratio
    #==============================================
    #============== Training ======================
    #==============================================
    device = next(model.parameters()).device
    
    best_metric = float('inf')  # Set to a large value
    avg_train_loss_history = []
    avg_test_loss_history = []
    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for source_train, target_train in train_loader:
            sources = source_train.permute(1, 0, 2).to(device) # Shape: (sequence_len, batch_size, features)
            targets = target_train.permute(1, 0, 2).to(device) # Shape: (sequence_len, batch_size, features)
            
            optimizer.zero_grad()
            outputs = model(
                src=sources, 
                trg=targets, 
                teacher_forcing_ratio=teacher_forcing_ratio
            )

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        #======================Test Loss=====================
        model.eval()  # Set the model to evaluation mode
        total_test_loss = 0
        with torch.no_grad():
            for source_test, targets_test in test_loader:
                sources = source_test.permute(1, 0, 2).to(device)
                targets = targets_test.permute(1, 0, 2).to(device)
                
                outputs = model(
                    src=sources, 
                    trg=targets, 
                    teacher_forcing_ratio=0.0
                )
                
                loss = criterion(outputs, targets)
                total_test_loss += loss.item()

            avg_test_loss = total_test_loss / len(test_loader)
            if verbose == True:
                print(f"Epoch [{epoch+1}/{num_epochs}], "
                    f"Train Loss: {avg_train_loss:.6f}, "
                    f"Validation Loss: {avg_test_loss:.6f}")
            
        avg_train_loss_history.append(avg_train_loss)
        avg_test_loss_history.append(avg_test_loss)

        if avg_test_loss < best_metric:
            best_metric = avg_test_loss
            if model_path is not None:
                model.save_checkpoint(model_path)

    return model, avg_train_loss_history, avg_test_loss_history

def prepareTraining(model, trainData, testData, trainingConfig: TrainingConfig, verbose=False):
    #==============================================
    #=============== Hyperparameters ==============
    #==============================================
    batch_size = trainingConfig.batch_size
    learning_rate = trainingConfig.learning_rate

    #==============================================
    #============== Create Dataloader =============
    #==============================================
    train_loader = createDataLoaders(
        batch_size=batch_size, dataset=trainData, shuffle=True
    )
    test_loader = createDataLoaders(
        batch_size=batch_size, dataset=testData, shuffle=False
    )
        
    #==============================================
    #============== Model Setup ===================
    #==============================================
    size_model = countModelParameters(model)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    #==============================================
    #============== Verbose ===================
    #==============================================
    if verbose:
        print(f"Size of train loader: {len(train_loader)}, Size of test loader: {len(test_loader)}")
        print(f"Size of model: {size_model}")
        print(model)

    return model, criterion, optimizer, train_loader, test_loader

def createDataLoaders(batch_size, dataset, shuffle=True):
    # Convert all input data into tensors and stack them
    tensor_list = [torch.stack([torch.from_numpy(d).float() for d in data]) for data in dataset]
    
    num_samples = tensor_list[0].shape[0]
    assert all(t.shape[0] == num_samples for t in tensor_list), "All input tensors must have the same number of samples."
    
    # Create a dataset
    dataset = TensorDataset(*tensor_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)

    return dataloader

# Calculate total parameters
def countModelParameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ..config import TrainingConfig
from ..models import CustomLossFunction

def prepareTraining(model, trainData, testData, trainingConfig: TrainingConfig, verbose=False):
    batch_size = trainingConfig.batch_size
    learning_rate = trainingConfig.learning_rate

    train_loader = createDataLoaders(
        batch_size=batch_size, dataset=trainData, shuffle=True
    )
    test_loader = createDataLoaders(
        batch_size=batch_size, dataset=testData, shuffle=False
    )
        
    size_model = countModelParameters(model)
    criterion = CustomLossFunction(
        lambda_trans=trainingConfig.lambda_transmission, 
        lambda_class=trainingConfig.lambda_traffic_class,
        lambda_context=trainingConfig.lambda_context)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if verbose:
        print(f"Size of train loader: {len(train_loader)}, Size of test loader: {len(test_loader)}")
        print(f"Used device: {next(model.parameters()).device}")
        print(f"Size of model: {size_model}")
        print(model)

    return criterion, optimizer, train_loader, test_loader


def trainModelHelper(model, criterion, optimizer, train_loader, test_loader, trainingConfig, verbose=False):
    num_epochs = trainingConfig.num_epochs
    #==============================================
    #============== Training ======================
    #==============================================
    best_metric = float('inf')  # Set to a large value
    avg_train_loss_history = []
    avg_test_loss_history = []
    best_model = None

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            sources, targets, last_trans_sources, _, traffics, traffics_class, transmissions, sourcesNoSmooth = (
                data.to(next(model.parameters()).device) for data in batch
            )
            sources = sources.permute(1, 0, 2)
            sourcesNoSmooth = sourcesNoSmooth.permute(1, 0, 2)
            targets = targets.permute(1, 0, 2)

            traffics_class = traffics_class.view(-1).to(torch.long)
            last_trans_sources = last_trans_sources.permute(1, 0, 2)
            
            optimizer.zero_grad()
            out_traffic, out_traffic_class, out_trans, out_target = model(sources, last_trans_sources, sourcesNoSmooth)
            loss, _ = criterion(
                out_traffic, traffics,
                out_traffic_class, traffics_class,
                out_trans, transmissions,
                out_target, targets
            )
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        #======================Test Loss=====================
        model.eval()
        total_test_loss = 0
        total_test_loss_traffic = 0
        with torch.no_grad():
            for batch in test_loader:
                sources, targets, last_trans_sources, _, traffics, traffics_class, transmissions, sourcesNoSmooth = (
                    data.to(next(model.parameters()).device) for data in batch
                )
                sources = sources.permute(1, 0, 2)
                sourcesNoSmooth = sourcesNoSmooth.permute(1, 0, 2)
                targets = targets.permute(1, 0, 2)
                traffics_class = traffics_class.view(-1).to(torch.long)
                last_trans_sources = last_trans_sources.permute(1, 0, 2)
                
                out_traffic, out_traffic_class, out_trans, out_target = model(sources, last_trans_sources, sourcesNoSmooth)
                loss, loss_traffic = criterion(
                    out_traffic, traffics,
                    out_traffic_class, traffics_class,
                    out_trans, transmissions,
                    out_target, targets
                )
                total_test_loss += loss.item()
                total_test_loss_traffic += loss_traffic.item()

            avg_test_loss = total_test_loss / len(test_loader)
            avg_test_loss_traffic = total_test_loss_traffic / len(test_loader)

            if verbose:
                print(f"Epoch [{epoch+1}/{num_epochs}], "
                      f"Train Loss: {avg_train_loss:.4f}, "
                      f"Validation Loss: {avg_test_loss:.4f}, "
                      f"Validation Loss (Traffic): {avg_test_loss_traffic:.4f}")
                
        if avg_test_loss < best_metric:
            bestWights = model.state_dict()  # Save model state
            best_metric = avg_test_loss

        avg_train_loss_history.append(avg_train_loss)
        avg_test_loss_history.append(avg_test_loss)
    
    return bestWights, avg_train_loss_history, avg_test_loss_history



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

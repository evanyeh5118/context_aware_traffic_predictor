import torch
from torch.utils.data import DataLoader, TensorDataset

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




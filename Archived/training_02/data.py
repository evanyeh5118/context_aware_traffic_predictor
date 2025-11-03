import torch
from torch.utils.data import DataLoader, TensorDataset


def createDataLoadersEnhanced(batch_size, dataset, shuffle=True, num_workers=4, pin_memory=True):
    """Create DataLoaders with enhanced performance options."""
    tensor_list = [torch.stack([torch.from_numpy(d).float() for d in data]) for data in dataset]

    num_samples = tensor_list[0].shape[0]
    assert all(t.shape[0] == num_samples for t in tensor_list), "All input tensors must have the same number of samples."

    dataset = TensorDataset(*tensor_list)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        prefetch_factor=2 if num_workers > 0 else None
    )

    return dataloader



import torch


def save_checkpoint(checkpoint_dir, model, epoch, optimizer, scheduler, val_loss, is_best):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }

    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()

    checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
    torch.save(checkpoint, checkpoint_path)

    if is_best:
        best_path = checkpoint_dir / "best_model.pt"
        torch.save(checkpoint, best_path)

    keep_last_n_checkpoints(checkpoint_dir, n=3, keep_best=True)


def keep_last_n_checkpoints(checkpoint_dir, n=3, keep_best=True):
    """Keep only the last n checkpoints plus the best one."""
    checkpoints = sorted(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    if len(checkpoints) > n:
        for old_checkpoint in checkpoints[:-n]:
            old_checkpoint.unlink()


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load training checkpoint."""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint['epoch'], checkpoint['val_loss']



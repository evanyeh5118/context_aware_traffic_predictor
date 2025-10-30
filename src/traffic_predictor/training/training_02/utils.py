import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def log_epoch_info(epoch, num_epochs, train_loss, val_loss, val_loss_components,
                   train_loss_components, num_batches, learning_rate_history,
                   is_best, patience_counter):
    """Log detailed epoch information."""
    current_lr = learning_rate_history[-1] if learning_rate_history else "N/A"

    msg = (
        f"Epoch [{epoch + 1}/{num_epochs}] | "
        f"Train Loss: {train_loss:.6f} | "
        f"Val Loss: {val_loss:.6f} | "
        f"LR: {current_lr}"
    )

    if is_best:
        msg += " ⭐ BEST"

    msg += f" | Patience: {patience_counter}"

    logger.info(msg)



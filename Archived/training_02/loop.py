import torch
import torch.nn.utils as utils

from .utils import logger, log_epoch_info
from .checkpoints import save_checkpoint


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device,
                   gradient_clip, use_mixed_precision):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    loss_components = {'traffic': 0, 'classification': 0, 'transmission': 0, 'context': 0}

    for batch_idx, batch in enumerate(train_loader):
        sources, targets, last_trans_sources, _, traffics, traffics_class, transmissions, sourcesNoSmooth = (
            data.to(device) for data in batch
        )
        sources = sources.permute(1, 0, 2)
        sourcesNoSmooth = sourcesNoSmooth.permute(1, 0, 2)
        targets = targets.permute(1, 0, 2)
        traffics_class = traffics_class.view(-1).to(torch.long)
        last_trans_sources = last_trans_sources.permute(1, 0, 2)

        optimizer.zero_grad(set_to_none=True)

        if use_mixed_precision:
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                out_traffic, out_traffic_class, out_trans, out_target = model(
                    sources, last_trans_sources, sourcesNoSmooth
                )
                loss, loss_components_batch = criterion(
                    out_traffic, traffics,
                    out_traffic_class, traffics_class,
                    out_trans, transmissions,
                    out_target, targets
                )
        else:
            out_traffic, out_traffic_class, out_trans, out_target = model(
                sources, last_trans_sources, sourcesNoSmooth
            )
            loss, loss_components_batch = criterion(
                out_traffic, traffics,
                out_traffic_class, traffics_class,
                out_trans, transmissions,
                out_target, targets
            )

        if not torch.isfinite(loss):
            logger.error(f"Non-finite loss detected at batch {batch_idx}: {loss.item()}")
            raise ValueError("Loss became infinite or NaN during training")

        if use_mixed_precision:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()

        total_loss += loss.item()

    return {'total': total_loss}, loss_components


def validate_model(model, val_loader, criterion, device, use_mixed_precision):
    """Validate the model."""
    model.eval()
    total_loss = 0
    loss_components = {}

    with torch.no_grad():
        for batch in val_loader:
            sources, targets, last_trans_sources, _, traffics, traffics_class, transmissions, sourcesNoSmooth = (
                data.to(device) for data in batch
            )
            sources = sources.permute(1, 0, 2)
            sourcesNoSmooth = sourcesNoSmooth.permute(1, 0, 2)
            targets = targets.permute(1, 0, 2)
            traffics_class = traffics_class.view(-1).to(torch.long)
            last_trans_sources = last_trans_sources.permute(1, 0, 2)

            if use_mixed_precision:
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    out_traffic, out_traffic_class, out_trans, out_target = model(
                        sources, last_trans_sources, sourcesNoSmooth
                    )
                    loss, loss_components_batch = criterion(
                        out_traffic, traffics,
                        out_traffic_class, traffics_class,
                        out_trans, transmissions,
                        out_target, targets
                    )
            else:
                out_traffic, out_traffic_class, out_trans, out_target = model(
                    sources, last_trans_sources, sourcesNoSmooth
                )
                loss, loss_components_batch = criterion(
                    out_traffic, traffics,
                    out_traffic_class, traffics_class,
                    out_trans, transmissions,
                    out_target, targets
                )

            total_loss += loss.item()

    return {'total': total_loss}, loss_components


def trainModelHelper(parameters, model, criterion, optimizer, scheduler, scaler,
                     device, train_loader, val_loader, checkpoint_dir, verbose=False):
    """Training loop with comprehensive features."""
    num_epochs = parameters['num_epochs']
    gradient_clip = parameters.get("gradient_clip", 1.0)
    early_stop_patience = parameters.get("early_stop_patience", 10)
    use_mixed_precision = parameters.get("use_mixed_precision", True)

    best_metric = float('inf')
    best_epoch = 0
    patience_counter = 0
    avg_train_loss_history = []
    avg_val_loss_history = []
    learning_rate_history = []

    loss_components_history = {
        'traffic': [],
        'classification': [],
        'transmission': [],
        'context': []
    }

    for epoch in range(num_epochs):
        train_losses, train_loss_components = train_one_epoch(
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            gradient_clip=gradient_clip,
            use_mixed_precision=use_mixed_precision
        )

        avg_train_loss = train_losses['total'] / len(train_loader)
        avg_train_loss_history.append(avg_train_loss)

        val_losses, val_loss_components = validate_model(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            use_mixed_precision=use_mixed_precision
        )

        avg_val_loss = val_losses['total'] / len(val_loader)
        avg_val_loss_history.append(avg_val_loss)

        for key in loss_components_history:
            if key in val_loss_components:
                avg_component_loss = val_loss_components[key] / len(val_loader)
                loss_components_history[key].append(avg_component_loss)

        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            learning_rate_history.append(current_lr)

        is_best = False
        if avg_val_loss < best_metric:
            is_best = True
            best_metric = avg_val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1

        if verbose:
            log_epoch_info(epoch, num_epochs, avg_train_loss, avg_val_loss,
                           val_loss_components, train_loss_components, len(val_loader),
                           learning_rate_history, is_best, patience_counter)

        if patience_counter >= early_stop_patience:
            if verbose:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
            break

        if parameters.get("save_checkpoints", True):
            save_checkpoint(
                checkpoint_dir=checkpoint_dir,
                model=model,
                epoch=epoch,
                optimizer=optimizer,
                scheduler=scheduler,
                val_loss=avg_val_loss,
                is_best=is_best
            )

    model.load_state_dict(best_model_state)

    histories = {
        'train_loss': avg_train_loss_history,
        'val_loss': avg_val_loss_history,
        'learning_rate': learning_rate_history,
        'loss_components': loss_components_history
    }

    training_info = {
        'best_epoch': best_epoch,
        'best_val_loss': best_metric,
        'total_epochs_trained': epoch + 1,
        'stopped_early': patience_counter >= early_stop_patience
    }

    return model, histories, training_info



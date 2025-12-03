import math
import logging
import time
from typing import Dict, Optional, List, Any, Tuple
from dataclasses import dataclass, field

import torch
import wandb
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    TrainerCallback,
    TrainerState,
    TrainerControl,
    TrainingArguments,
    Trainer,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingCycleMetrics:
    """Container for metrics collected during a training cycle"""
    train_losses: List[float] = field(default_factory=list)
    val_losses: List[float] = field(default_factory=list)
    loss_momentum: Optional[float] = None
    cycle_start_loss: Optional[float] = None
    mean_train_loss: Optional[float] = None
    mean_val_loss: Optional[float] = None
    best_cycle_val_loss: Optional[float] = None
    volatility: float = 0.0
    overfit_ratio: float = 1.0
    gen_gap: float = 0.0

    def reset(self):
        """Reset all metrics for a new cycle"""
        self.train_losses = []
        self.val_losses = []
        self.cycle_start_loss = None
        self.mean_train_loss = None
        self.mean_val_loss = None
        self.best_cycle_val_loss = None
        self.volatility = 0.0
        self.overfit_ratio = 1.0
        self.gen_gap = 0.0

    def compute_statistics(self, momentum_factor: float = 0.7):
        """Compute statistical metrics based on collected losses"""
        logger.info('Calculating stats')
        if not self.train_losses:
            return

        # Training loss statistics
        self.mean_train_loss = sum(self.train_losses) / len(self.train_losses)

        # Compute volatility
        if len(self.train_losses) > 1:
            variance = sum((x - self.mean_train_loss) ** 2 for x in self.train_losses) / len(self.train_losses)
            self.volatility = math.sqrt(variance) / max(self.mean_train_loss, 1e-10)

        # Validation loss statistics
        if self.val_losses:
            self.best_cycle_val_loss = min(self.val_losses)
            self.mean_val_loss = sum(self.val_losses) / len(self.val_losses)

            # Generalization gap
            self.gen_gap = self.mean_val_loss - self.mean_train_loss

            # Overfitting ratio
            if self.mean_train_loss > 1e-8:
                self.overfit_ratio = (self.mean_val_loss + 1e-8) / max(self.mean_train_loss, 1e-8)
            else:
                self.overfit_ratio = float('inf')


class CycleAwareLRScheduler:
    """Learning rate scheduler that maintains its own step counter independent of global steps"""

    def __init__(
        self,
        optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        min_lr: float = 1e-6,
        true_zero_warmup: bool = False,
        cycle_end_decay_factor: float = 1.0
    ):
        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.cycle_end_decay_factor = cycle_end_decay_factor
        self.min_lr = min_lr
        self.true_zero_warmup = true_zero_warmup

        # Initialize step counter
        self.step_count = 0

        # Store base learning rates
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def state_dict(self):
        """Returns the state of the scheduler as a dict."""
        return {
            'step_count': self.step_count,
            'base_lrs': self.base_lrs,
        }

    def load_state_dict(self, state_dict):
        """Loads the scheduler state."""
        self.step_count = state_dict['step_count']
        self.base_lrs = state_dict['base_lrs']

    def get_last_lr(self):
        """Return current learning rates - required by Trainer"""
        return [group['lr'] for group in self.optimizer.param_groups]

    def step(self):
        """
        Advance the scheduler state by one step with comprehensive error handling.
        """
        try:
            # Increment the counter only if it hasn't reached the maximum.
            if self.step_count < self.num_training_steps:
                self.step_count += 1
            else:
                self.step_count = self.num_training_steps  # Clamp only when necessary

            # Update learning rates for all parameter groups based on the current step.
            for i, param_group in enumerate(self.optimizer.param_groups):
                try:
                    param_group['lr'] = self._get_lr(i)
                    logger.debug(f"[LR SCHEDULER] step_count={self.step_count}, lr={param_group['lr']:.8f}")
                except Exception as e:
                    logger.error(f"Error updating LR for group {i}: {e}")
                    param_group['lr'] = self.min_lr  # Safe fallback
        except Exception as e:
            logger.error(f"Error in scheduler step: {e}")
            # Ensure all groups have valid LR as fallback
            for group in self.optimizer.param_groups:
                group['lr'] = self.min_lr

    def _get_lr(self, group_idx):
        """
        Get learning rate for parameter group index with comprehensive safety checks.
        """
        try:
            base_lr = self.base_lrs[group_idx]
        except (IndexError, TypeError):
            logger.error(f"Invalid group_idx {group_idx} for base_lrs of length {len(self.base_lrs)}")
            return self.min_lr  # Safe fallback

        current_step = self.step_count

        # Safety checks for scheduler parameters
        if self.num_training_steps <= 0:
            logger.warning("Invalid training steps (<=0). Using safe fallback.")
            return self.min_lr

        if self.num_warmup_steps < 0:
            logger.warning("Invalid warmup steps (<0). Using 0 instead.")
            self.num_warmup_steps = 0

        # Compute LR based on phase (warmup or decay)
        try:
            if current_step < self.num_warmup_steps:
                # Warmup phase - can start from true zero or min_lr
                warmup_steps = max(1, self.num_warmup_steps)  # Ensure non-zero denominator
                if self.true_zero_warmup:
                    lr = base_lr * (current_step / warmup_steps)  # True zero start
                else:
                    lr = self.min_lr + (base_lr - self.min_lr) * (current_step / warmup_steps)
            else:
                # Cosine decay phase
                denominator = max(1, self.num_training_steps - self.num_warmup_steps)  # Ensure non-zero denominator
                progress = (current_step - self.num_warmup_steps) / denominator
                progress = min(1.0, progress)  # Ensure never exceeds 1.0
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

                # Apply end-of-cycle decay factor if in last 10% of cycle
                if progress > 0.9:
                    end_progress = (progress - 0.9) / 0.1
                    end_decay = 1.0 + (self.cycle_end_decay_factor - 1.0) * end_progress
                    cosine_decay *= end_decay

                lr = self.min_lr + (base_lr - self.min_lr) * cosine_decay
        except Exception as e:
            logger.error(f"Error computing learning rate: {e}")
            return self.min_lr  # Safe fallback

        # Final safety check for valid LR
        if not isinstance(lr, (int, float)) or math.isnan(lr) or math.isinf(lr) or lr <= 0:
            logger.warning(f"Invalid LR calculated: {lr}. Using minimum LR instead.")
            return self.min_lr

        return lr

    def reset(self):
        """Reset step counter to zero, for new cycle"""
        self.step_count = 0

    def get_lr(self):
        """Return current learning rates"""
        return [group['lr'] for group in self.optimizer.param_groups]

    def set_base_lrs(self, new_base_lrs):
        """Update base LRs for each parameter group"""
        if len(new_base_lrs) != len(self.base_lrs):
            raise ValueError("Length of new base LRs doesn't match number of parameter groups")
        self.base_lrs = new_base_lrs


class StochasticWeightAveraging(TrainerCallback):
    """
    Implements Stochastic Weight Averaging (SWA) to find flatter minima.
    Starts averaging model weights after a specified percentage of training.
    """
    def __init__(
        self,
        start_pct: float = 0.75,
        update_every: int = 10,
        use_equal_weights: bool = False
    ):
        self.start_pct = start_pct
        self.update_every = update_every
        self.use_equal_weights = use_equal_weights
        self.swa_state = None
        self.swa_count = 0
        self.swa_started = False

    def on_step_begin(self, args, state, control, **kwargs):
        # Check if we should start SWA
        total_steps = args.max_steps if args.max_steps > 0 else args.num_train_epochs * args.steps_per_epoch
        swa_start_step = int(total_steps * self.start_pct)

        if state.global_step >= swa_start_step and not self.swa_started:
            logger.info(f"Starting Stochastic Weight Averaging at step {state.global_step}")
            self.swa_started = True

        # Update SWA weights
        if self.swa_started and state.global_step % self.update_every == 0:
            self._update_swa_weights(kwargs.get('model', None) or state.model)

    def _update_swa_weights(self, model):
        """Update the SWA weights with the current model weights."""
        model_module = model.module if hasattr(model, "module") else model

        if self.swa_state is None:
            # First time - initialize SWA state
            self.swa_state = {}
            for name, param in model_module.named_parameters():
                if param.requires_grad:
                    self.swa_state[name] = param.data.clone().cpu()
            self.swa_count = 1
            logger.info(f"Initialized SWA with parameters from step {self.swa_count}")
        else:
            # Update existing SWA state
            if self.use_equal_weights:
                # Equal weighting for all models (simple average)
                n = self.swa_count
                for name, param in model_module.named_parameters():
                    if name in self.swa_state and param.requires_grad:
                        self.swa_state[name].mul_(n / (n + 1)).add_(param.data.cpu(), alpha=1 / (n + 1))
            else:
                # Exponential moving average
                alpha = 0.75  # Smoothing factor
                for name, param in model_module.named_parameters():
                    if name in self.swa_state and param.requires_grad:
                        self.swa_state[name].mul_(alpha).add_(param.data.cpu(), alpha=(1 - alpha))

            self.swa_count += 1
            if self.swa_count % 10 == 0:
                logger.info(f"Updated SWA weights (count: {self.swa_count})")

    def on_train_end(self, args, state, control, **kwargs):
        """Apply SWA weights to model at the end of training."""
        if self.swa_state is None or self.swa_count < 2:
            logger.info("SWA not activated or insufficient updates - skipping final model update")
            return

        logger.info(f"Applying final SWA weights from {self.swa_count} models")
        model = kwargs.get('model', None) or state.model
        model_module = model.module if hasattr(model, "module") else model

        with torch.no_grad():
            for name, param in model_module.named_parameters():
                if name in self.swa_state and param.requires_grad:
                    param.copy_(self.swa_state[name].to(param.device))

        logger.info("SWA weights successfully applied to model")


class AdaptiveGradientCallback(TrainerCallback):
    """
    Combines gradient normalization and noise injection into a single adaptive callback.
    Automatically determines when to apply each technique based on training dynamics.
    """
    def __init__(
        self,
        # Common parameters
        enable_normalization: bool = True,
        enable_noise: bool = True,
        log_frequency: int = 50,

        # Normalization parameters
        norm_factor: float = 1.0,
        clip_threshold: Optional[float] = None,
        per_layer: bool = False,
        exclude_bias: bool = True,

        # Noise parameters
        base_noise_scale: float = 0.01,
        max_noise_scale: float = 0.05,
        memory_window_size: int = 10,
        memorization_threshold: float = 1.5,
        validation_batch_frequency: int = 50,
        ema_decay: float = 0.9,

        # Adaptive control
        auto_adjust: bool = True,
        monitor_window_size: int = 100
    ):
        # Store parameters
        self.enable_normalization = enable_normalization
        self.enable_noise = enable_noise
        self.log_frequency = log_frequency

        # Normalization parameters
        self.norm_factor = norm_factor
        self.clip_threshold = clip_threshold
        self.per_layer = per_layer
        self.exclude_bias = exclude_bias

        # Noise parameters
        self.base_noise_scale = base_noise_scale
        self.max_noise_scale = max_noise_scale
        self.memory_window_size = memory_window_size
        self.memorization_threshold = memorization_threshold
        self.validation_batch_frequency = validation_batch_frequency
        self.ema_decay = ema_decay

        # Adaptive parameters
        self.auto_adjust = auto_adjust
        self.monitor_window_size = monitor_window_size

        # Tracking variables
        self.step_count = 0
        self.last_logged_step = -1

        # Training dynamics tracking
        self.recent_losses = []
        self.loss_volatility = 0.0
        self.grad_norm_history = []

        # Memorization tracking
        self.param_train_grad_ema = {}
        self.param_val_grad_ema = {}
        self.memorization_scores = {}

        # Validation data
        self.val_dataloader = None
        self.val_iter = None

    def on_train_begin(self, args, state, control, **kwargs):
        """Initialize validation dataloader iterator if noise is enabled"""
        if not self.enable_noise:
            return

        if not hasattr(args, 'eval_dataset') or args.eval_dataset is None:
            logger.warning("AdaptiveGradientCallback needs eval_dataset for noise injection, but none was provided")
            self.enable_noise = False
            return

        try:
            # Get validation dataloader from trainer
            self.val_dataloader = kwargs.get('train_dataloader').__class__(
                args.eval_dataset,
                batch_size=kwargs.get('train_dataloader').batch_size,
                collate_fn=kwargs.get('train_dataloader').collate_fn
            )
            self.val_iter = iter(self.val_dataloader)
            logger.info(f"AdaptiveGradientCallback initialized with validation dataloader")
        except Exception as e:
            logger.error(f"Failed to initialize validation dataloader: {e}")
            self.enable_noise = False

    def on_step_begin(self, args, state, control, **kwargs):
        """Track step count for scheduling and monitoring"""
        self.step_count += 1

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Track training dynamics through loss values"""
        if logs is None or not logs or "loss" not in logs or math.isnan(logs["loss"]):
            return

        # Store recent losses for volatility calculation
        self.recent_losses.append(logs["loss"])
        if len(self.recent_losses) > self.monitor_window_size:
            self.recent_losses.pop(0)

        # Calculate loss volatility when we have enough data
        if len(self.recent_losses) >= 5:
            mean_loss = sum(self.recent_losses) / len(self.recent_losses)
            variance = sum((x - mean_loss) ** 2 for x in self.recent_losses) / len(self.recent_losses)
            self.loss_volatility = math.sqrt(variance) / max(mean_loss, 1e-10)

    def _get_validation_batch(self):
        """Get the next validation batch, reinitializing iterator if needed"""
        if self.val_dataloader is None:
            return None

        try:
            # Try to get next batch
            batch = next(self.val_iter)
        except StopIteration:
            # Reinitialize iterator and try again
            self.val_iter = iter(self.val_dataloader)
            try:
                batch = next(self.val_iter)
            except StopIteration:
                logger.error("Validation dataset appears to be empty")
                return None

        # Move batch to correct device if needed
        device = next(self.trainer.model.parameters()).device
        batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}

        return batch

    def _update_grad_ema(self, model, validation_batch=None):
        """Update gradient EMAs and memorization scores for noise injection"""
        if not self.enable_noise:
            return

        is_validation = validation_batch is not None
        ema_dict = self.param_val_grad_ema if is_validation else self.param_train_grad_ema

        # For validation, perform forward/backward pass without affecting model
        if is_validation:
            # Store current model state
            optimizer_state = self.trainer.optimizer.state_dict()

            # Forward/backward pass on validation batch
            self.trainer.model.zero_grad()
            with torch.enable_grad():  # Ensure we get gradients
                outputs = self.trainer.model(**validation_batch)
                if isinstance(outputs, dict):
                    loss = outputs["loss"] if "loss" in outputs else outputs.loss
                else:
                    loss = outputs.loss
                loss.backward()

            # Restore optimizer state after validation grad computation
            self.trainer.optimizer.load_state_dict(optimizer_state)

        # Update EMA for each parameter's gradient
        for name, param in model.named_parameters():
            if param.grad is None:
                continue

            # Compute normalized absolute gradient
            grad_norm = torch.norm(param.grad.abs()).item() / torch.norm(param.data).item()

            # Initialize EMA for this parameter if needed
            if name not in ema_dict:
                ema_dict[name] = grad_norm
            else:
                ema_dict[name] = self.ema_decay * ema_dict[name] + (1 - self.ema_decay) * grad_norm

            # Update memorization score if we have both training and validation gradients
            if not is_validation and name in self.param_val_grad_ema and name in self.param_train_grad_ema:
                if self.param_val_grad_ema[name] > 1e-10:  # Avoid division by zero
                    # Memorization = ratio of training gradient to validation gradient
                    ratio = self.param_train_grad_ema[name] / self.param_val_grad_ema[name]

                    if name not in self.memorization_scores:
                        self.memorization_scores[name] = ratio
                    else:
                        # Smooth update of memorization score
                        self.memorization_scores[name] = 0.8 * self.memorization_scores[name] + 0.2 * ratio

    def _apply_gradient_noise(self, model):
        """Apply noise to parameters based on memorization scores"""
        if not self.enable_noise or not self.memorization_scores:
            return 0, 0

        params_modified = 0
        total_params = 0

        with torch.no_grad():
            for name, param in model.named_parameters():
                total_params += 1
                if param.grad is None or name not in self.memorization_scores:
                    continue

                # Get memorization score and determine if it's high enough for noise
                memo_score = self.memorization_scores[name]
                if memo_score > self.memorization_threshold:
                    # Scale noise based on memorization score
                    # More memorization = more noise
                    severity = min(3.0, (memo_score - self.memorization_threshold) / self.memorization_threshold)
                    noise_scale = min(
                        self.max_noise_scale,
                        self.base_noise_scale * (1.0 + severity)
                    )

                    # Scale by parameter norm for stability
                    param_norm = torch.norm(param.data)
                    scaled_noise = torch.randn_like(param.grad) * noise_scale * param_norm

                    # Add noise to gradient
                    param.grad.add_(scaled_noise)
                    params_modified += 1

        return params_modified, total_params

    def _apply_gradient_normalization(self, model):
        """Apply gradient normalization based on training dynamics"""
        if not self.enable_normalization:
            return 0.0, 0.0, 0, 0

        # Track gradient statistics for logging
        total_norm_before = 0.0
        total_norm_after = 0.0
        params_modified = 0
        total_params = 0

        # Auto-adjust norm factor based on volatility
        current_norm_factor = self.norm_factor
        if self.auto_adjust and self.loss_volatility > 0:
            # Stronger normalization for higher volatility
            volatility_adjustment = min(2.0, 1.0 + self.loss_volatility)
            current_norm_factor *= volatility_adjustment

        if self.per_layer:
            # Per-layer normalization
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue

                # Skip bias parameters if configured
                if self.exclude_bias and 'bias' in name:
                    continue

                total_params += 1

                # Calculate current gradient norm
                grad_norm = param.grad.norm().item()
                total_norm_before += grad_norm ** 2

                # Skip if gradient is already small
                if grad_norm < 1e-8:
                    continue

                with torch.no_grad():
                    # Scale gradient to the target norm
                    scale = current_norm_factor / grad_norm

                    # Apply optional clipping
                    if self.clip_threshold is not None:
                        scale = min(scale, self.clip_threshold / grad_norm)

                    # Apply scaling
                    if scale != 1.0:
                        param.grad.mul_(scale)
                        params_modified += 1

                # Track new norm after scaling
                total_norm_after += param.grad.norm().item() ** 2
        else:
            # Global normalization across all parameters
            # First pass: calculate total gradient norm
            all_grads = []
            for name, param in model.named_parameters():
                if param.grad is None:
                    continue

                # Skip bias parameters if configured
                if self.exclude_bias and 'bias' in name:
                    continue

                # Focus on LoRA parameters if we're doing LoRA fine-tuning
                if 'lora' not in name.lower() and not param.requires_grad:
                    continue

                total_params += 1
                all_grads.append(param.grad)
                total_norm_before += param.grad.norm().item() ** 2

            total_norm_before = total_norm_before ** 0.5

            # Skip if the total gradient is too small
            if total_norm_before < 1e-8 or not all_grads:
                return total_norm_before, total_norm_before, 0, total_params

            # Calculate scaling factor
            scale = current_norm_factor / total_norm_before

            # Apply optional clipping
            if self.clip_threshold is not None:
                scale = min(scale, self.clip_threshold / total_norm_before)

            # Second pass: apply scaling
            if scale != 1.0:
                with torch.no_grad():
                    for grad in all_grads:
                        grad.mul_(scale)
                        params_modified += 1

                # Calculate new norm
                total_norm_after = 0.0
                for param in model.parameters():
                    if param.grad is not None:
                        total_norm_after += param.grad.norm().item() ** 2
                total_norm_after = total_norm_after ** 0.5
            else:
                total_norm_after = total_norm_before

        return total_norm_before, total_norm_after, params_modified, total_params


    def on_pre_optimizer_step(self, args, state, control, model=None, optimizer=None, **kwargs):
        """Apply adaptive gradient modifications before optimizer step"""
        if model is None or optimizer is None:
            return

        # Get current model
        model = model.module if hasattr(model, "module") else model

        # Periodically compute validation gradients for noise injection
        if self.enable_noise and self.step_count % self.validation_batch_frequency == 0:
            validation_batch = self._get_validation_batch()
            if validation_batch is not None:
                self._update_grad_ema(model, validation_batch)

        # Update training gradient EMA for noise
        if self.enable_noise:
            self._update_grad_ema(model)

        # Adaptively decide whether to apply techniques based on training dynamics
        use_norm = self.enable_normalization
        use_noise = self.enable_noise

        if self.auto_adjust:
            # Use normalization more aggressively with high volatility
            use_norm = self.enable_normalization and (self.loss_volatility > 0.08 or self.step_count < 100)

            # Use noise more when memorization is likely (later in training)
            if self.step_count < 100:
                use_noise = False
            else:
                use_noise = self.enable_noise

            # Track if loss is flatlining
            if len(self.recent_losses) > 50:
                recent_improvement = self.recent_losses[0] - self.recent_losses[-1]
                relative_improvement = recent_improvement / max(abs(self.recent_losses[0]), 1e-10)

                # If almost no improvement in last 50 steps, force noise to be used
                if relative_improvement < 0.01:  # Less than 1% improvement
                    use_noise = self.enable_noise
                    logger.info(f"Detected loss plateau (improvement={relative_improvement:.4f}), forcing noise")

        # Apply techniques in correct order: first noise, then normalization
        noise_params_modified, noise_total_params = 0, 0
        if use_noise:
            noise_params_modified, noise_total_params = self._apply_gradient_noise(model)

        norm_before, norm_after, norm_params_modified, norm_total_params = 0.0, 0.0, 0, 0
        if use_norm:
            norm_before, norm_after, norm_params_modified, norm_total_params = self._apply_gradient_normalization(model)

        # Store gradient norms for tracking
        self.grad_norm_history.append(norm_before)
        if len(self.grad_norm_history) > self.monitor_window_size:
            self.grad_norm_history.pop(0)

        # Log progress periodically
        if self.step_count % self.log_frequency == 0 and self.step_count > self.last_logged_step:
            log_message = []

            if use_norm:
                log_message.append(f"Gradient norm: {norm_before:.4f} → {norm_after:.4f}, "
                                 f"Modified {norm_params_modified}/{norm_total_params} parameters")

            if use_noise:
                log_message.append(f"Gradient noise: Applied to {noise_params_modified}/{noise_total_params} parameters")

            if log_message:
                logger.info(" | ".join(log_message))

            # Additional metrics
            if self.auto_adjust:
                logger.info(f"Training dynamics: Loss volatility={self.loss_volatility:.4f}, "
                           f"Using norm={use_norm}, Using noise={use_noise}")

            # Try to log to wandb if available
            try:
                if wandb.run is not None:
                    log_dict = {
                        "adaptive_grad/loss_volatility": self.loss_volatility,
                        "adaptive_grad/using_normalization": 1 if use_norm else 0,
                        "adaptive_grad/using_noise": 1 if use_noise else 0
                    }

                    if use_norm:
                        log_dict.update({
                            "adaptive_grad/norm_before": norm_before,
                            "adaptive_grad/norm_after": norm_after,
                            "adaptive_grad/norm_ratio": norm_after / (norm_before + 1e-8),
                            "adaptive_grad/norm_params_modified": norm_params_modified,
                        })

                    if use_noise:
                        log_dict.update({
                            "adaptive_grad/noise_params_modified": noise_params_modified,
                            "adaptive_grad/noise_params_pct": noise_params_modified / max(1, noise_total_params) * 100,
                            "adaptive_grad/max_memorization": max(self.memorization_scores.values(), default=0)
                        })

                    wandb.log(log_dict, step=state.global_step)
            except Exception:
                pass

            self.last_logged_step = self.step_count

class CoordinatedDropoutCallback(TrainerCallback):
    """
    Callback that dynamically adjusts LoRA dropout based on generalization gap during training.

    This callback monitors the difference between training and evaluation loss to detect
    overfitting or underfitting, and adjusts dropout rates accordingly to improve generalization.
    """

    def __init__(
        self,
        trainer,
        initial_dropout: float = 0.1,
        gap_threshold: float = 0.10,
        max_dropout: float = 0.5,
        min_dropout: float = 0.0,
        dropout_step: float = 0.05,
        improvement_threshold: float = 0.15,
        min_loss: float = 0.01,
    ):
        """
        Initialize the dropout callback.

        Args:
            trainer: The Trainer instance
            initial_dropout: Starting dropout value
            gap_threshold: When gap < -threshold, increase dropout (for overfitting)
            max_dropout: Maximum allowed dropout value
            min_dropout: Minimum allowed dropout value
            dropout_step: Amount to adjust dropout on each update
            improvement_threshold: Threshold for determining underfitting
            min_loss: Minimum loss value to consider for underfitting detection
        """
        self.trainer = trainer
        self.current_dropout = initial_dropout
        self.gap_threshold = gap_threshold
        self.max_dropout = max_dropout
        self.min_dropout = min_dropout
        self.dropout_step = dropout_step
        self.improvement_threshold = improvement_threshold
        self.min_loss = min_loss

        # Training state tracking
        self.last_train_loss = None
        self.last_eval_loss = None
        self.action_history = []

        logger.info(f"Initialized dropout callback with current_dropout={initial_dropout}")

    def adjust_dropout(self, increase: bool) -> bool:
        """Adjusts dropout rate up or down within bounds."""
        logger.info("ADJUSTING DROPOUT")
        old_dropout = self.current_dropout

        # Calculate new dropout value
        if increase:
            if self.current_dropout >= self.max_dropout:
                return False
            new_dropout = min(
                self.current_dropout + self.dropout_step,
                self.max_dropout
            )
        else:
            if self.current_dropout <= self.min_dropout:
                return False
            new_dropout = max(
                self.current_dropout - self.dropout_step,
                self.min_dropout
            )

        # If there's no change, exit early
        if abs(new_dropout - old_dropout) < 1e-9:
            return False

        # Update the class-level dropout tracker
        self.current_dropout = new_dropout

        modules_updated = self._update_model_dropouts(new_dropout)

        direction = "Increased" if increase else "Decreased"
        if modules_updated > 0:
            logger.info(
                f"{direction} LoRA dropout from {old_dropout:.3f} to {new_dropout:.3f} "
                f"in {modules_updated} modules"
            )
            return True
        else:
            logger.warning("No dropout modules were updated!")
            return False

    def _update_model_dropouts(self, new_dropout: float) -> int:
        """
        Update all LoRA dropout modules in the model.

        Returns:
            int: Number of modules updated
        """
        modules_updated = 0

        # Try to get the model structure
        if not hasattr(self.trainer, 'model'):
            logger.warning("Trainer has no model attribute")
            return 0

        # Log model type for debugging
        logger.info(f"Model type: {type(self.trainer.model)}")

        # Check if model follows the base_model.model.model.layers pattern (like in your case)
        try:
            model_has_layers = False
            layers = None

            # Navigate to layers through different possible paths
            if hasattr(self.trainer.model, 'base_model'):
                base_model = self.trainer.model.base_model

                if hasattr(base_model, 'model') and hasattr(base_model.model, 'model'):
                    inner_model = base_model.model.model

                    if hasattr(inner_model, 'layers'):
                        layers = inner_model.layers
                        model_has_layers = True
                        logger.info(f"Found {len(layers)} layers")

            # If we found a layers attribute, iterate through each layer
            if model_has_layers and layers is not None:
                for i, layer in enumerate(layers):
                    layer_modules_updated = 0

                    # Try to find common attention and MLP modules by checking all attributes
                    # This is more flexible than hardcoding specific names
                    for attr_name in dir(layer):
                        # Skip private attributes and non-module attributes
                        if attr_name.startswith('_') or not hasattr(layer, attr_name):
                            continue

                        attr = getattr(layer, attr_name)

                        # Process modules recursively
                        if isinstance(attr, torch.nn.Module):
                            layer_modules_updated += self._process_module_recursively(attr, new_dropout)

                    if layer_modules_updated > 0:
                        logger.info(f"Updated {layer_modules_updated} modules in layer {i}")
                        modules_updated += layer_modules_updated
        except Exception as e:
            logger.warning(f"Error while navigating model structure: {e}")

        # If no modules updated using the layer approach, try a direct module traversal
        if modules_updated == 0:
            logger.info("Trying direct module traversal approach")
            for name, module in self.trainer.model.named_modules():
                if hasattr(module, 'lora_dropout'):
                    if self._update_lora_dropout(module, new_dropout):
                        modules_updated += 1

        return modules_updated

    def _process_module_recursively(self, module, new_dropout):
        """
        Process a module and its children recursively to find and update LoRA dropouts.

        Returns:
            int: Number of modules updated
        """
        modules_updated = 0

        # Check if this module has lora_dropout attribute
        if hasattr(module, 'lora_dropout'):
            if self._update_lora_dropout(module, new_dropout):
                modules_updated += 1

        # Process all child modules
        for name, child_module in module.named_children():
            modules_updated += self._process_module_recursively(child_module, new_dropout)

        return modules_updated

    def _update_lora_dropout(self, module, new_dropout):
        """
        Update the dropout in a single module.

        Handles different possible storage formats for the lora_dropout attribute:
        - Direct torch.nn.Dropout module
        - torch.nn.ModuleDict containing adapters
        - Other container types

        Returns:
            bool: True if dropout was updated, False otherwise
        """
        if not hasattr(module, 'lora_dropout'):
            return False

        lora_dropout = module.lora_dropout
        updated = False

        # Case 1: ModuleDict of adapters (common in newer PEFT versions)
        if isinstance(lora_dropout, torch.nn.ModuleDict):
            logger.info(f"ModuleDict keys: {list(lora_dropout.keys())}")

            for adapter_name, adapter_module in lora_dropout.items():
                logger.info(f"Examining adapter '{adapter_name}' of type {type(adapter_module)}")

                if isinstance(adapter_module, torch.nn.Dropout):
                    old_p = adapter_module.p
                    adapter_module.p = new_dropout
                    logger.info(f"Updated dropout for adapter '{adapter_name}' from {old_p:.3f} to {new_dropout:.3f}")
                    updated = True
                elif isinstance(adapter_module, torch.nn.Identity) and new_dropout > 0:
                    logger.info(f"Replacing Identity with Dropout({new_dropout:.3f}) for adapter '{adapter_name}'")
                    lora_dropout[adapter_name] = torch.nn.Dropout(p=new_dropout)
                    updated = True

        # Case 2: Direct Dropout module
        elif isinstance(lora_dropout, torch.nn.Dropout):
            old_p = lora_dropout.p
            module.lora_dropout.p = new_dropout
            logger.info(f"Updated dropout module directly from {old_p:.3f} to {new_dropout:.3f}")
            updated = True

        # Case 3: Identity that should be replaced with Dropout
        elif isinstance(lora_dropout, torch.nn.Identity) and new_dropout > 0:
            logger.info(f"Replacing Identity with Dropout({new_dropout:.3f})")
            module.lora_dropout = torch.nn.Dropout(p=new_dropout)
            updated = True

        # Case 4: Other container type, try to find Dropout modules inside
        elif hasattr(lora_dropout, '__iter__'):
            try:
                for i, submodule in enumerate(lora_dropout):
                    if isinstance(submodule, torch.nn.Dropout):
                        old_p = submodule.p
                        submodule.p = new_dropout
                        logger.info(f"Updated dropout in container at index {i} from {old_p:.3f} to {new_dropout:.3f}")
                        updated = True
            except (TypeError, AttributeError):
                logger.warning(f"Container could not be iterated: {type(lora_dropout)}")

        # Case 5: Unknown type
        else:
            logger.warning(f"Unknown lora_dropout type: {type(lora_dropout)}")

        return updated

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Dict[str, float] = None,
        **kwargs,
    ):
        """Track training loss when logged."""
        if logs and "loss" in logs and not math.isnan(logs["loss"]):
            self.last_train_loss = logs["loss"]


    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs,
    ):
        """
        Evaluate the generalization gap and adjust dropout if needed.
        This is called after each evaluation phase during training.
        """
        logger.info(f"Dropout callback on_evaluate called with metrics: {metrics}")

        if not metrics or "eval_loss" not in metrics:
            return

        current_eval_loss = metrics["eval_loss"]
        self.last_eval_loss = current_eval_loss

        # If we haven't seen a training loss yet, we can't calculate the gap
        if self.last_train_loss is None:
            logger.warning("No training loss available yet, skipping dropout adjustment")
            return

        if math.isnan(current_eval_loss) or math.isnan(self.last_train_loss):
            return

        # Calculate generalization gap (train_loss - eval_loss)
        # Negative gap means eval_loss > train_loss (potential overfitting)
        gap = self.last_train_loss - current_eval_loss

        logger.info(
            f"Eval step {state.global_step}: Generalization gap = {gap:.4f} "
            f"(train_loss={self.last_train_loss:.4f}, eval_loss={current_eval_loss:.4f})"
        )


        # Detect overfitting (large negative gap)
        if gap < -self.gap_threshold:
            logger.info(f"Detected overfitting (gap={gap:.4f})")
            if self.adjust_dropout(increase=True):
                # Log adjustment action
                try:
                    if wandb.run is not None:
                        wandb.log({"dropout_action": "increased"}, step=state.global_step)
                except ImportError:
                    pass

        # Detect underfitting (small or negative gap with high loss)
        elif (gap > -self.improvement_threshold and
              self.last_train_loss > self.min_loss):
            logger.info(f"Detected potential underfitting (gap={gap:.4f}, loss still high)")
            if self.adjust_dropout(increase=False):
                # Log adjustment action
                try:
                    if wandb.run is not None:
                        wandb.log({"dropout_action": "decreased"}, step=state.global_step)
                except ImportError:
                    pass
        try:
            wandb.log({"lora_dropout": self.current_dropout})
            # Try multiple approaches to access wandb
            if 'wandb' in globals():
                if wandb.run is not None:
                    wandb.log({"dropout/lora_dropout": self.current_dropout})
                    wandb.log({"dropout/generalization_gap": gap})
                    logger.info(f"Logged current dropout value {self.current_dropout} to wandb")
        except Exception as e:
            logger.warning(f"Failed to log dropout to wandb: {e}")


class EMAEvaluationCallback(TrainerCallback):
    """
    Callback to evaluate both standard and EMA models during training.

    This callback hooks into the evaluation process to:
    1. Track the best model (standard or EMA) based on evaluation loss
    2. Save checkpoints of the best model
    3. Push to Hugging Face Hub if configured
    """
    # Class-level flag to prevent re-entrance
    _in_evaluation = False

    def __init__(
        self,
        output_dir: str = None,
        use_ema: bool = True,
        ema_decay: float = 0.999,
        max_retries: int = 3,
        checkpoint = None,
        trainer = None
    ):
        self.output_dir = output_dir
        self.current_cycle = 0
        self.trainer = trainer  # Store trainer directly from initialization
        self.logger = logging.getLogger(__name__)

        # Create or use provided model checkpoint manager
        if checkpoint is not None:
            self.checkpoint = checkpoint
        else:
            self.checkpoint = ModelCheckpoint(
                use_ema=use_ema,
                ema_decay=ema_decay,
                max_retries=max_retries
            )

        # Set trainer on checkpoint if provided
        if self.trainer is not None and self.checkpoint is not None:
            self.checkpoint.trainer = self.trainer
            self.logger.info("Trainer explicitly provided and set on checkpoint manager")

    def on_train_begin(self, args, state, control, **kwargs):
        """Initialize callback when training begins."""
        self.logger.info("Initializing EMA Evaluation Callback")

        # If trainer wasn't provided at init, try to get it now
        if self.trainer is None:
            # Get trainer from kwargs (if available)
            trainer = kwargs.get('trainer', None)

            # If not in kwargs, try to get it via state
            if trainer is None and hasattr(state, 'trainer'):
                trainer = state.trainer

            # If still not found, try to get it from control
            if trainer is None and hasattr(control, 'trainer'):
                trainer = control.trainer

            # Store trainer reference for later use
            self.trainer = trainer

            if self.trainer is None:
                self.logger.warning("Could not find trainer in on_train_begin - EMA evaluation may not work")
            else:
                self.logger.info("Found trainer in on_train_begin")

                # Set trainer on checkpoint manager
                if self.checkpoint is not None:
                    self.checkpoint.trainer = self.trainer
        else:
            self.logger.info("Using trainer provided during initialization")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """
        Run evaluation on both standard and EMA models.

        Called after the trainer's evaluation is complete.
        """
        # Check if we're already in an evaluation to prevent recursion
        if EMAEvaluationCallback._in_evaluation:
            return

        try:
            # Set the flag to prevent re-entrance
            EMAEvaluationCallback._in_evaluation = True

            if metrics is None or "eval_loss" not in metrics:
                self.logger.warning("No evaluation metrics found")
                return

            # We need to use our stored reference to the trainer
            if self.trainer is None:
                self.logger.warning("No trainer available - cannot evaluate EMA model")
                return

            # Get model directly from trainer instead of state
            model = self.trainer.model
            if model is None:
                self.logger.warning("No model available from trainer")
                return

            # Update cycle count (assuming evaluation happens at the end of each cycle)
            self.current_cycle += 1

            # Log the standard model loss before EMA evaluation
            self.logger.info(f"Standard model eval loss: {metrics['eval_loss']:.6f}")

            # Use our checkpoint manager to evaluate both standard and EMA models
            saved_new_best = self.checkpoint.on_evaluate(
                model=model,
                trainer=self.trainer,
                cycle=self.current_cycle,
                metrics=metrics
            )

            # Log to wandb
            if wandb.run is not None:
                try:
                    wandb.log({
                        "cycle": self.current_cycle,
                        "eval_loss": metrics["eval_loss"],
                        "best_eval_loss": self.checkpoint.best_loss,
                        "best_model_is_ema": self.checkpoint.best_is_ema
                    })

                    if saved_new_best:
                        wandb.run.summary["best_loss"] = self.checkpoint.best_loss
                        wandb.run.summary["best_cycle"] = self.checkpoint.best_cycle
                        wandb.run.summary["best_is_ema"] = self.checkpoint.best_is_ema
                except Exception as e:
                    self.logger.warning(f"Failed to log to wandb: {e}")
        finally:
            # Always reset the flag, even if there's an exception
            EMAEvaluationCallback._in_evaluation = False

    def on_train_end(self, args, state, control, **kwargs):
        """Called at the end of training."""
        # Log final best model info
        if hasattr(self.checkpoint, 'best_loss'):
            self.logger.info(f"Training completed. Best model was {'EMA' if self.checkpoint.best_is_ema else 'standard'} "
                            f"from cycle {self.checkpoint.best_cycle} with loss {self.checkpoint.best_loss:.6f}")
        else:
            self.logger.warning("Training completed but no best model was tracked")


class ModelCheckpoint:
    """Manages model checkpoints with proper EMA handling and dynamic decay."""
    def __init__(self, use_ema: bool = True, ema_decay: float = 0.999, max_retries: int = 3,
                 ema_min_decay: float = 0.90, ema_warmup_steps: int = 300, lora: bool = True,
                 enable_hub_push: bool = False):
        # State tracking
        self.best_loss = float('inf')
        self.best_state = None
        self.best_cycle = None
        self.best_is_ema = False

        # EMA configuration
        self.use_ema = use_ema
        self.ema_decay = ema_decay  # Target maximum decay
        self.ema_min_decay = ema_min_decay  # Starting minimum decay
        self.ema_warmup_steps = ema_warmup_steps  # Steps to approach target decay
        self.ema_state = None
        self.ema_updates_count = 0  # Counter for number of EMA updates

        # LoRA configuration
        self.lora = lora  # Whether to only track LoRA parameters or all parameters

        # Hub configuration
        self.max_retries = max_retries
        self.enable_hub_push = enable_hub_push
        self.trainer = None

        self.logger = logging.getLogger(__name__)

    def _safe_push_to_hub(self, model, commit_message: str, attempt: int = 0) -> bool:
        """Safely push to hub with retries for transient errors."""
        # Check if hub push is enabled to avoid conflicts with external push
        if not self.enable_hub_push:
            return False
            
        if attempt >= self.max_retries:
            self.logger.warning("Max retries exceeded for hub push - skipping")
            return False

        try:
            from huggingface_hub import HfApi
            hf_api = HfApi()

            self.logger.info(f"Pushing model to hub: {self.trainer.args.hub_model_id}")
            model.push_to_hub(
                self.trainer.args.hub_model_id,
                commit_message=commit_message,
                config_kwargs={"skip_metadata": True},
                create_model_card=False
            )
            self.logger.info(f"Successfully pushed model to hub with message: {commit_message}")
            return True
        except Exception as e:
            error_msg = str(e).lower()
            if "504" in error_msg or "timeout" in error_msg:
                self.logger.warning(f"Hub push attempt {attempt + 1} failed with timeout - will retry")
                import time
                time.sleep(5 * (attempt + 1))  # Exponential backoff
                return self._safe_push_to_hub(model, commit_message, attempt + 1)
            else:
                self.logger.error(f"Hub push failed with error: {str(e)}")
                return False

    def _get_model_state(self, model):
        """Extract trainable parameters from model based on lora setting."""
        model_module = model.module if hasattr(model, "module") else model
        state_dict = {}

        if self.lora:
            # LoRA mode: Only save adapter parameters and lm_head
            self.logger.info("LoRA mode: Capturing only LoRA parameters and lm_head")

            # Try to get adapter state dict for LoRA or other adapters
            try:
                if hasattr(model_module, "get_adapter_state_dict"):
                    state_dict.update(model_module.get_adapter_state_dict())
                    self.logger.info("Retrieved adapter state dict")
                else:
                    for name, param in model_module.named_parameters():
                        if "lora" in name.lower() or param.requires_grad:
                            state_dict[name] = param.data.cpu().clone()
            except Exception as e:
                self.logger.warning(f"Failed to get adapter state dict: {e}. Collecting parameters manually.")
                for name, param in model_module.named_parameters():
                    if param.requires_grad:
                        state_dict[name] = param.data.cpu().clone()

            # Save LM head if it exists (often critical for performance)
            if hasattr(model_module, "lm_head"):
                state_dict["lm_head.weight"] = model_module.lm_head.weight.data.cpu().clone()
                if hasattr(model_module.lm_head, "bias") and model_module.lm_head.bias is not None:
                    state_dict["lm_head.bias"] = model_module.lm_head.bias.data.cpu().clone()
        else:
            # Full model mode: Save all parameters
            self.logger.info("Full model mode: Capturing all parameters")
            for name, param in model_module.named_parameters():
                state_dict[name] = param.data.cpu().clone()

        self.logger.info(f"Captured {len(state_dict)} parameters")
        return state_dict

    def _update_ema_state(self, current_state):
        """Update EMA state using current model state with dynamic decay."""
        if self.ema_state is None:
            # First time - just clone the current state
            self.ema_state = {k: v.clone() for k, v in current_state.items()}
            self.ema_updates_count = 0
            self.logger.info(f"Initialized EMA state with {len(self.ema_state)} tensors")
        else:
            # Increment update counter
            self.ema_updates_count += 1

            # Calculate dynamic decay that starts lower and increases over time
            # Using a smooth exponential approach to the target decay
            warmup_progress = min(1.0, self.ema_updates_count / self.ema_warmup_steps)
            decay_range = self.ema_decay - self.ema_min_decay
            effective_decay = self.ema_min_decay + decay_range * (1 - math.exp(-5 * warmup_progress))

            # Update EMA weights with new values
            for k, v in current_state.items():
                if k in self.ema_state:
                    self.ema_state[k] = self.ema_state[k] * effective_decay + v * (1 - effective_decay)
                else:
                    self.ema_state[k] = v.clone()

            self.logger.info(f"Updated EMA state with decay {effective_decay:.6f} (update #{self.ema_updates_count})")

        return self.ema_state

    def _apply_weights(self, model, weights):
        """Apply weights to model based on lora setting."""
        model_module = model.module if hasattr(model, "module") else model

        if self.lora:
            # Only apply LoRA and lm_head parameters
            for name, param in model_module.named_parameters():
                if name in weights and ("lora" in name.lower() or param.requires_grad):
                    param.data.copy_(weights[name].to(param.device))

            # Apply weights to LM head if it exists
            if hasattr(model_module, "lm_head") and "lm_head.weight" in weights:
                model_module.lm_head.weight.data.copy_(weights["lm_head.weight"].to(model_module.lm_head.weight.device))
                if hasattr(model_module.lm_head, "bias") and model_module.lm_head.bias is not None and "lm_head.bias" in weights:
                    model_module.lm_head.bias.data.copy_(weights["lm_head.bias"].to(model_module.lm_head.bias.device))
        else:
            # Apply all parameters
            for name, param in model_module.named_parameters():
                if name in weights:
                    param.data.copy_(weights[name].to(param.device))

    def on_evaluate(self, model, trainer, cycle, metrics=None):
        """
        Evaluate both regular and EMA models, tracking the best of either.

        Args:
            model: The current model
            trainer: The trainer object
            cycle: Current training cycle number
            metrics: Current evaluation metrics if available

        Returns:
            bool: Whether a new best model was saved
        """
        if metrics is None or "eval_loss" not in metrics:
            self.logger.warning("No eval_loss found in metrics, skipping checkpoint")
            return False

        standard_loss = metrics["eval_loss"]
        self.logger.info(f"Standard model eval loss: {standard_loss:.6f}")

        # Get current model state
        current_state = self._get_model_state(model)

        # Track if we saved a new best model
        saved_new_best = False

        # Handle EMA if enabled
        ema_loss = float('inf')
        if self.use_ema:
            # Update EMA state with dynamic decay
            self._update_ema_state(current_state)

            # Store original weights
            original_weights = {k: v.clone() for k, v in current_state.items()}

            # Apply EMA weights to model
            self._apply_weights(model, self.ema_state)

            # Evaluate with EMA weights
            try:
                # Run evaluation
                ema_eval_output = trainer.evaluate()
                ema_loss = ema_eval_output.get("eval_loss", float('inf'))
                self.logger.info(f"EMA model eval loss: {ema_loss:.6f}")
            except Exception as e:
                self.logger.error(f"Error evaluating EMA model: {e}")
                ema_loss = float('inf')

            # Restore original weights
            self._apply_weights(model, original_weights)

        # Determine which model is better and if it beats the current best
        current_is_ema = self.use_ema and ema_loss < standard_loss
        current_best_loss = ema_loss if current_is_ema else standard_loss

        # If this is better than our best so far, update best state
        if current_best_loss < self.best_loss:
            self.best_loss = current_best_loss
            self.best_cycle = cycle
            self.best_is_ema = current_is_ema

            # Store appropriate state
            if current_is_ema:
                self.best_state = {k: v.clone() for k, v in self.ema_state.items()}
                self.logger.info(f"New best model is EMA with loss {ema_loss:.6f}")
            else:
                self.best_state = {k: v.clone() for k, v in current_state.items()}
                self.logger.info(f"New best model is standard with loss {standard_loss:.6f}")

            saved_new_best = True

            # Push to hub if available
            if hasattr(model, 'push_to_hub') and self.trainer is not None:
                if hasattr(self.trainer.args, 'hub_model_id') and self.trainer.args.hub_model_id:
                    # Apply best state to model for pushing
                    if current_is_ema:
                        # Need to apply EMA weights before pushing
                        temp_weights = {k: v.clone() for k, v in current_state.items()}
                        self._apply_weights(model, self.best_state)

                    # Push to hub
                    model_type = "EMA" if current_is_ema else "standard"
                    commit_message = f"NOM"
                    push_success = self._safe_push_to_hub(model, commit_message)

                    # Restore original weights if needed
                    if current_is_ema:
                        self._apply_weights(model, temp_weights)

                    if push_success:
                        self.logger.info(f"Pushed new best model to hub: {model_type} with loss {self.best_loss:.6f}")

        # Log which model performed better this cycle
        if self.use_ema:
            if ema_loss < standard_loss:
                self.logger.info(f"EMA model performed better: {ema_loss:.6f} vs {standard_loss:.6f}")
            else:
                self.logger.info(f"Standard model performed better: {standard_loss:.6f} vs {ema_loss:.6f}")

        # Log overall best model so far
        self.logger.info(f"Best overall model: {'EMA' if self.best_is_ema else 'standard'} "
                         f"from cycle {self.best_cycle} with loss {self.best_loss:.6f}")

        return saved_new_best

class AdaptiveCycleCosineLRCallback(TrainerCallback):
    """
    Advanced learning rate scheduler with cycle-based training, backtracking,
    and adaptive learning rate and cycle length adjustments based on training dynamics.
    """
    def __init__(
        self,
        trainer,
        output_dir: str,
        # Cycle parameters
        initial_cycle_length: int = 300,
        min_cycle_length: int = 100,
        max_cycle_length: int = 3000,
        base_cycle_growth_factor: float = 1.0,
        adaptive_cycle_growth: float = 1.2,

        # Learning rate parameters
        initial_lr: float = 5e-5,
        min_lr: float = 1e-6,
        warmup_ratio: float = 0.1,
        true_zero_warmup: bool = False,
        cycle_end_decay_factor: float = 0.8,

        # Early stopping parameters
        max_cycles_without_improvement: int = 3,

        # Training dynamics parameters
        momentum_factor: float = 0.7,
        volatility_penalty: float = 0.2,
        generalization_threshold: float = 0.15,

        # Backtracking parameters
        enable_backtracking: bool = True,
        max_consecutive_backtracks: int = 3,
        overfit_ratio_threshold: float = 1.4,
        validation_regression_threshold: float = 1.05,

        # Experimental features
        use_ema: bool = False,
        ema_decay: float = 0.999,
        sharpness_aware: bool = False,
        sharpness_aware_factor: float = 0.05,

        # Implementation details
        max_retries: int = 3,
        existing_checkpoint = False,
        lora: bool = True,
        enable_hub_push: bool = False
    ):
        # Store parameters
        self.trainer = trainer
        self.output_dir = output_dir

        # Cycle parameters
        self.current_cycle_length = initial_cycle_length
        self.min_cycle_length = min_cycle_length
        self.max_cycle_length = max_cycle_length
        self.base_cycle_growth_factor = base_cycle_growth_factor
        self.adaptive_cycle_growth = adaptive_cycle_growth

        # Learning rate parameters
        self.current_max_lr = initial_lr
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.warmup_ratio = warmup_ratio
        self.true_zero_warmup = true_zero_warmup
        self.cycle_end_decay_factor = cycle_end_decay_factor

        # Early stopping parameters
        self.max_cycles_without_improvement = max_cycles_without_improvement

        # Training dynamics parameters
        self.momentum_factor = momentum_factor
        self.volatility_penalty = volatility_penalty
        self.generalization_threshold = generalization_threshold

        # Backtracking parameters
        self.enable_backtracking = enable_backtracking
        self.max_consecutive_backtracks = max_consecutive_backtracks
        self.overfit_ratio_threshold = overfit_ratio_threshold
        self.validation_regression_threshold = validation_regression_threshold

        # Experimental features
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.sharpness_aware = sharpness_aware
        self.sharpness_aware_factor = sharpness_aware_factor

        # Implementation details
        self.max_retries = max_retries

        # Internal state
        self.current_cycle = 0
        self.cycles_without_improvement = 0
        self.step_in_cycle = 0
        self.cycle_boundaries = {}
        self.best_overall_val_loss = float('inf')
        self.consecutive_backtrack_count = 0
        self.backtrack_performed = False

        # Metrics for current cycle
        self.metrics = TrainingCycleMetrics()

        # History tracking
        self.best_val_loss_per_cycle = {}
        self.metric_history = {
            'cycle_lengths': [],
            'max_lr_per_cycle': [],
            'loss_volatility': [],
            'gen_gaps': [],
            'steps': []
        }

        if existing_checkpoint:
            self.checkpoint = existing_checkpoint
        else:
            self.checkpoint = ModelCheckpoint(use_ema=use_ema, ema_decay=ema_decay, max_retries=max_retries, lora=lora, enable_hub_push=enable_hub_push)
        self.checkpoint.trainer = trainer

        # Initialize optimizer state and LR scheduler (defer until training begins)
        self.optimizer_initial_state = None
        self.lr_scheduler = None

        logger.info(f"Initialized AdaptiveCycleCosineLR: initial_cycle={initial_cycle_length}, "
                   f"initial_lr={initial_lr:.2e}")

        # Log experimental features
        if any([true_zero_warmup, use_ema, sharpness_aware]):
            exp_features = []
            if true_zero_warmup:
                exp_features.append("true zero warmup")
            if use_ema:
                exp_features.append(f"EMA (decay={ema_decay})")
            if sharpness_aware:
                exp_features.append(f"sharpness-aware (factor={sharpness_aware_factor})")

            logger.info(f"Enabled experimental features: {', '.join(exp_features)}")

    def _safe_push_to_hub(self, model, commit_message: str, attempt: int = 0) -> bool:
        """Safely push to hub with retries for transient errors."""
        # Check if hub push is enabled via checkpoint manager
        if hasattr(self.checkpoint, 'enable_hub_push') and not self.checkpoint.enable_hub_push:
            return False
            
        if attempt >= self.max_retries:
            logger.warning("Max retries exceeded for hub push - skipping")
            return False

        try:
            from huggingface_hub import HfApi
            hf_api = HfApi()

            logger.info(f"Pushing model to hub: {self.trainer.args.hub_model_id}")
            model.push_to_hub(
                self.trainer.args.hub_model_id,
                commit_message=commit_message,
                config_kwargs={"skip_metadata": True},
                create_model_card=False
            )
            logger.info(f"Successfully pushed model to hub with message: {commit_message}")
            return True
        except Exception as e:
            error_msg = str(e).lower()
            if "504" in error_msg or "timeout" in error_msg:
                logger.warning(f"Hub push attempt {attempt + 1} failed with timeout - will retry")
                import time
                time.sleep(5 * (attempt + 1))  # Exponential backoff
                return self._safe_push_to_hub(model, commit_message, attempt + 1)
            else:
                logger.error(f"Hub push failed with error: {str(e)}")
                return False

    def _save_optimizer_state(self):
        """Save initial optimizer state for potential momentum reset."""
        if self.trainer is None or self.trainer.optimizer is None:
            logger.warning("Trainer or optimizer not available - deferring optimizer state save")
            return
            
        try:
            self.optimizer_initial_state = {
                'state': self.trainer.optimizer.state_dict()['state'].copy(),
                'param_groups': [pg.copy() for pg in self.trainer.optimizer.param_groups]
            }
            logger.info("Saved initial optimizer state")
        except Exception as e:
            logger.warning(f"Failed to save optimizer state: {e}")
            self.optimizer_initial_state = None

    def _reset_optimizer_momentum(self):
        """Reset optimizer momentum buffers to initial state but preserve learning rates."""
        if self.optimizer_initial_state is None:
            logger.warning("No initial optimizer state available - skipping momentum reset")
            return
            
        try:
            current_lrs = [pg['lr'] for pg in self.trainer.optimizer.param_groups]

            # Reset momentum for each parameter
            for group_idx, param_group in enumerate(self.trainer.optimizer.param_groups):
                for p in param_group['params']:
                    param_id = p.data_ptr()
                    if param_id in self.optimizer_initial_state['state']:
                        self.trainer.optimizer.state[p] = {
                            k: v.clone() if torch.is_tensor(v) else v
                            for k, v in self.optimizer_initial_state['state'][param_id].items()
                        }

            # Restore current learning rates (don't overwrite with initial rates)
            for group_idx, lr in enumerate(current_lrs):
                self.trainer.optimizer.param_groups[group_idx]['lr'] = lr

            logger.info("Reset optimizer momentum buffers for new cycle")
        except Exception as e:
            logger.warning(f"Failed to reset optimizer momentum: {e}")

    def _add_gradient_noise(self):
        """
        This method has been removed to avoid redundancy with the GradientNoiseCallback.
        Use the dedicated GradientNoiseCallback instead for gradient noise injection.
        """
        pass

    def _create_lr_scheduler(self):
        """Create a new learning rate scheduler for the current cycle."""
        if self.trainer is None or self.trainer.optimizer is None:
            logger.warning("Trainer or optimizer not available - deferring LR scheduler creation")
            return
            
        optimizer = self.trainer.optimizer

        # Safety check for cycle length
        if self.current_cycle_length <= 0:
            logger.warning(f"Invalid cycle length: {self.current_cycle_length}. Setting to minimum safe value.")
            self.current_cycle_length = max(1, self.min_cycle_length)

        # Calculate warmup steps
        num_warmup = int(self.current_cycle_length * self.warmup_ratio)
        num_warmup = max(0, num_warmup)  # Ensure non-negative

        logger.info(f"Creating LR scheduler: warmup_steps={num_warmup}, "
                   f"total_steps={self.current_cycle_length}, max_lr={self.current_max_lr:.6f}")

        try:
            self.lr_scheduler = CycleAwareLRScheduler(
                optimizer=optimizer,
                num_warmup_steps=num_warmup,
                num_training_steps=self.current_cycle_length,
                cycle_end_decay_factor=self.cycle_end_decay_factor,
                min_lr=self.min_lr,
                true_zero_warmup=self.true_zero_warmup
            )
        except Exception as e:
            logger.error(f"Failed to create CycleAwareLRScheduler: {e}")
            # Create a fallback scheduler with safe values
            logger.warning("Creating fallback scheduler with safe values")
            self.current_cycle_length = max(1, self.current_cycle_length)
            num_warmup = 0
            self.lr_scheduler = CycleAwareLRScheduler(
                optimizer=optimizer,
                num_warmup_steps=num_warmup,
                num_training_steps=self.current_cycle_length,
                cycle_end_decay_factor=1.0,
                min_lr=self.min_lr,
                true_zero_warmup=False
            )

        # Update base learning rates
        for i in range(len(self.lr_scheduler.base_lrs)):
            old_base_lr = self.lr_scheduler.base_lrs[i]
            self.lr_scheduler.base_lrs[i] = self.current_max_lr
            logger.info(f"Updated base LR for param group {i}: {old_base_lr:.6f} -> {self.current_max_lr:.6f}")

        # Update optimizer learning rates to starting point for this cycle
        for group_idx, group in enumerate(optimizer.param_groups):
            old_lr = group['lr']
            if self.true_zero_warmup and num_warmup > 0:
                new_lr = 0.0  # True zero start
            else:
                new_lr = max(self.min_lr, self.current_max_lr * self.warmup_ratio)  # Standard warmup start

            group['lr'] = new_lr
            logger.info(f"Set optimizer LR for group {group_idx}: {old_lr:.6f} -> {new_lr:.6f}")

        # Update cycle boundaries
        if self.metric_history['steps']:
            start_step = self.metric_history['steps'][-1]
        else:
            start_step = 0

        end_step = start_step + self.current_cycle_length
        self.cycle_boundaries[self.current_cycle] = (start_step, end_step)

        # Reset cycle state
        self.step_in_cycle = 0
        self.metrics.reset()

        if self.current_cycle > 0:
            self._reset_optimizer_momentum()

        # Initialize scheduler's step counter
        if self.lr_scheduler is not None:
            self.lr_scheduler.step_count = 0

        logger.info(f"Cycle {self.current_cycle}: length={self.current_cycle_length}, "
                   f"max_lr={self.current_max_lr:.2e}, warmup={num_warmup} steps")

        # Set trainer's scheduler to our scheduler
        if self.lr_scheduler is not None:
            self.trainer.lr_scheduler = self.lr_scheduler

    def _apply_sharpness_aware_update(self):
        """
        Apply a sharpness-aware update step to find flatter minima.
        Based on the SAM optimizer concept but simplified for use with existing optimizers.
        """
        if not self.sharpness_aware or self.step_in_cycle % 10 != 0:
            return
        logger.info('applying sharpness aware')

        # Store original parameters
        original_params = {}
        for name, param in self.trainer.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                original_params[name] = param.data.clone()

        # Compute perturbed weights based on gradients
        with torch.no_grad():
            for name, param in self.trainer.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # Calculate perturbation scale proportional to parameter norm
                    scale = self.sharpness_aware_factor * (param.data.norm() /
                                                         (param.grad.norm() + 1e-12))
                    # Apply perturbation in gradient direction
                    param.add_(param.grad, alpha=scale)

        # Train for one step with perturbed weights to measure sharpness
        self.trainer.training_step(self.trainer.model, None)

        # Restore original weights
        with torch.no_grad():
            for name, param in self.trainer.model.named_parameters():
                if name in original_params:
                    param.copy_(original_params[name])

        logger.debug(f"Applied sharpness-aware update at step {self.step_in_cycle}")


    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Callback for the end of evaluation"""
        if metrics and "eval_loss" in metrics:
            eval_loss = metrics["eval_loss"]
            logger.info(f"Captured validation loss: {eval_loss:.6f}")
            self.metrics.val_losses.append(eval_loss)

    def on_train_begin(self, args, state, control, **kwargs):
        """Callback for the start of training"""
        logger.info("AdaptiveCycleCosineLR: Training started")

        # Now that training has begun, initialize optimizer state and LR scheduler
        self._save_optimizer_state()
        self._create_lr_scheduler()

        # Patch the trainer's get_learning_rate method
        try:
            def patched_get_lr():
                return self.trainer.optimizer.param_groups[0]['lr']
            self.trainer._get_learning_rate = patched_get_lr
        except Exception as e:
            logger.warning(f"Could not patch trainer's get_learning_rate method: {e}")

        # Make sure trainer is using our scheduler
        self.trainer.lr_scheduler = self.lr_scheduler

        # Initialize cycle and step counters
        self.current_cycle = 0
        self.step_in_cycle = 0

    def on_step_begin(self, args, state, control, **kwargs):
        """Callback for the start of a training step"""
        try:
            if self.lr_scheduler is None:
                logger.debug("LR scheduler not initialized yet - skipping step")
                return

            # Increment step counter
            self.step_in_cycle += 1

            if self.step_in_cycle >= self.current_cycle_length:
                logger.info(f"Cycle step ({self.step_in_cycle}) has reached cycle length ({self.current_cycle_length})")
                self._end_current_cycle(state, control)
                return  # Skip the rest of this method after ending the cycle

            # Log current LR periodically
            if self.step_in_cycle % 10 == 0 or self.step_in_cycle == 1:
                logger.info(f'logging stuff to do with cycles {self.current_cycle_length}')
                current_lr = self.trainer.optimizer.param_groups[0]['lr']
                cycle_progress = self.step_in_cycle / self.current_cycle_length

                logger.info(f"Step {state.global_step} (cycle step {self.step_in_cycle}): "
                          f"LR={current_lr:.2e}, cycle={self.current_cycle}, "
                          f"progress={cycle_progress:.1%}")

                # Log to wandb if available
                try:
                    if wandb.run is not None:
                        wandb.log({
                            "adaptive_lr/current_lr": current_lr,
                            "adaptive_lr/cycle_progress": cycle_progress
                        }, step=state.global_step)
                except Exception as e:
                    logger.debug(f"Failed to log to wandb: {e}")

            # Apply sharpness-aware update if enabled
            if self.sharpness_aware:
                self._apply_sharpness_aware_update()

        except Exception as e:
            logger.error(f"Error in on_step_begin: {e}")
            # Don't reraise the exception, just log it and continue
            import traceback
            logger.error(traceback.format_exc())

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Callback for logging events - primarily used to track training loss"""
        # Safety check for logs being None or not containing 'loss'
        if logs is None or not logs:
            return

        if "loss" not in logs or math.isnan(logs["loss"]):
            return

        loss = logs["loss"]

        # Track cycle start loss if not already set
        if self.metrics.cycle_start_loss is None:
            self.metrics.cycle_start_loss = loss

        # Store loss for this step
        self.metrics.train_losses.append(loss)

        # Update exponential moving average of loss
        if self.metrics.loss_momentum is None:
            self.metrics.loss_momentum = loss
        else:
            self.metrics.loss_momentum = (self.momentum_factor * self.metrics.loss_momentum +
                                         (1 - self.momentum_factor) * loss)

        # Log learning rate periodically
        if self.step_in_cycle % 10 == 0 or self.step_in_cycle == 1:
            try:
                # Ensure optimizer param_groups exists and has elements
                if hasattr(self.trainer, 'optimizer') and self.trainer.optimizer is not None:
                    if hasattr(self.trainer.optimizer, 'param_groups') and self.trainer.optimizer.param_groups:
                        current_lr = self.trainer.optimizer.param_groups[0]['lr']

                        # Try to log to wandb if available
                        try:
                            if wandb.run is not None:
                                wandb.log({
                                    "adaptive_lr/current_lr": current_lr,
                                    "adaptive_lr/cycle_progress": self.step_in_cycle / max(1, self.current_cycle_length)
                                }, step=state.global_step)
                        except Exception as e:
                            logger.debug(f"Failed to log to wandb: {e}")
            except Exception as e:
                logger.debug(f"Error in learning rate logging: {e}")

    def _get_current_cycle(self, global_step):
        """Determine which cycle a global step belongs to"""
        for cycle, (start, end) in self.cycle_boundaries.items():
            if start <= global_step < end:
                return cycle
        return self.current_cycle

    def _end_current_cycle(self, state, control):
        """
        Process the end of a training cycle by analyzing metrics,
        determining next cycle parameters, and deciding whether to backtrack.
        """
        # Safety check for early stopping
        if control.should_training_stop:
            logger.info("Control already signaled to stop training - skipping cycle transition")
            return

        # Increment cycle counter
        self.current_cycle += 1

        # Reset step counter
        self.step_in_cycle = 0

        # Reset scheduler for new cycle
        if self.lr_scheduler is not None:
            self.lr_scheduler.reset()
        else:
            logger.warning("LR scheduler not available for reset")

        # Set optimizer LR to the warmup starting point
        if self.true_zero_warmup:
            initial_lr = 0.0  # True zero start
        else:
            initial_lr = max(self.min_lr, self.current_max_lr * self.warmup_ratio)

        for group in self.trainer.optimizer.param_groups:
            group['lr'] = initial_lr

        # Take initial step in scheduler
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        else:
            logger.warning("LR scheduler not available for initial step")

        logger.info(f"============ ENDING CYCLE {self.current_cycle - 1} ============")
        logger.info(f"Global step: {state.global_step}, cycle_length: {self.current_cycle_length}")

        # Skip analysis if no training losses recorded
        if not self.metrics.train_losses:
            logger.warning(f"Cycle {self.current_cycle - 1} ended with no recorded training losses")
            return

        # Compute metrics for this cycle
        self.metrics.compute_statistics(self.momentum_factor)

        # Log training statistics
        mean_train_loss = self.metrics.mean_train_loss
        volatility = self.metrics.volatility
        logger.info(f"Training loss: mean={mean_train_loss:.6f}, volatility={volatility:.4f}")

        # Log validation statistics if available
        if self.metrics.val_losses:
            best_cycle_val_loss = self.metrics.best_cycle_val_loss
            mean_val_loss = self.metrics.mean_val_loss
            logger.info(f"Val losses: {self.metrics.val_losses}")
            logger.info(f"Best val loss (this cycle) = {best_cycle_val_loss:.6f}, Mean val loss = {mean_val_loss:.6f}")
        else:
            logger.info("No validation losses recorded this cycle.")
            best_cycle_val_loss = float('inf')
            mean_val_loss = float('inf')

        # Check for improvement over best overall validation loss
        if (self.metrics.best_cycle_val_loss is not None and
            self.metrics.best_cycle_val_loss <= self.best_overall_val_loss):
            improvement = (self.best_overall_val_loss - self.metrics.best_cycle_val_loss) / self.best_overall_val_loss
            logger.info(f"New overall best validation loss: {self.metrics.best_cycle_val_loss:.6f} (improvement: {improvement:.2%})")
            self.best_overall_val_loss = self.metrics.best_cycle_val_loss
            self.cycles_without_improvement = 0
        else:
            self.cycles_without_improvement += 1
            logger.info(f"No improvement vs. overall best. Cycles without improvement: {self.cycles_without_improvement}")

        # Check for overfitting
        overfit_ratio = self.metrics.overfit_ratio
        severe_overfitting = (overfit_ratio > self.overfit_ratio_threshold)
        logger.info(f"Overfit ratio = {overfit_ratio:.4f} (threshold={self.overfit_ratio_threshold:.2f}), severe_overfitting={severe_overfitting}")

        # Check for validation loss regression
        validation_regression = False
        if self.metrics.best_cycle_val_loss is not None and self.metrics.best_cycle_val_loss > self.best_overall_val_loss * self.validation_regression_threshold:
            validation_regression = True
            regression_percentage = ((self.metrics.best_cycle_val_loss / self.best_overall_val_loss) - 1) * 100
            logger.info(f"Validation loss regression detected: current={self.metrics.best_cycle_val_loss:.6f}, "
                       f"best={self.best_overall_val_loss:.6f} (worse by {regression_percentage:.2f}%)")

        # Check if we should stop due to no improvement
        if self.cycles_without_improvement >= self.max_cycles_without_improvement:
            logger.info(f"Early stopping after {self.cycles_without_improvement} cycles without improvement")
            control.should_training_stop = True
            # Set valid values for next cycle to avoid crashes
            self.current_cycle_length = max(1, self.current_cycle_length)
            self._create_lr_scheduler()
            return

        # Determine if we need to backtrack or apply a partial penalty
        backtracking_triggered = False

        if self.enable_backtracking:
            # Only attempt backtracking if we haven't exceeded the consecutive limit
            if self.consecutive_backtrack_count < self.max_consecutive_backtracks:
                # Check multiple conditions for backtracking
                should_backtrack = False
                backtrack_reason = ""

                if severe_overfitting:
                    should_backtrack = True
                    backtrack_reason = "severe overfitting"
                elif validation_regression and self.cycles_without_improvement >= 1:
                    should_backtrack = True
                    backtrack_reason = "validation loss regression"
                elif self.cycles_without_improvement >= 2:
                    should_backtrack = True
                    backtrack_reason = "multiple cycles without improvement"

                if should_backtrack:
                    logger.info(f"Backtracking triggered due to: {backtrack_reason}")
                    # Attempt backtracking if we have a valid model state from a previous cycle
                    if self.checkpoint.best_cycle is not None and self.checkpoint.best_cycle < self.current_cycle - 1:
                        model = self.trainer.model.module if hasattr(self.trainer.model, "module") else self.trainer.model
                        backtracking_triggered = self.checkpoint.restore(model)

                        if backtracking_triggered:
                            logger.info(f"Backtracking successful - restored model from cycle {self.checkpoint.best_cycle}")
                            # For backtracking, shorten the next cycle and reduce LR further
                            next_cycle_length = int(self.current_cycle_length * 0.7)
                            next_cycle_length = max(self.min_cycle_length, next_cycle_length)

                            # Check if cycle length is too small to continue
                            if next_cycle_length <= 10:
                                logger.info(f"Cycle length after backtracking ({next_cycle_length}) is too small. Signaling to end training.")
                                control.should_training_stop = True
                                next_cycle_length = max(1, next_cycle_length)

                            # Apply learning rate penalty after backtracking
                            backtrack_lr_factor = 0.4 ** (self.consecutive_backtrack_count + 1)
                            self.current_max_lr *= backtrack_lr_factor
                            logger.info(f"Reduced LR by {1-backtrack_lr_factor:.2%} to {self.current_max_lr:.6f}")

                            # Increase consecutive backtrack counter
                            self.consecutive_backtrack_count += 1
                            self.backtrack_performed = True

                            # Reset metrics
                            self.metrics.reset()
                            self.current_cycle_length = next_cycle_length
                            self._create_lr_scheduler()
                            return  # End processing after backtracking
                    else:
                        logger.info("Backtracking triggered but no valid model state available from earlier cycle.")
            else:
                logger.warning(f"Reached maximum consecutive backtracking limit ({self.max_consecutive_backtracks}). "
                              f"Continuing without backtracking.")

        # If we didn't backtrack but have issues, apply a partial penalty
        if not backtracking_triggered and (self.cycles_without_improvement > 0 or validation_regression):
            # For no improvement or regression, drop the LR and shorten the cycle without model reversion
            lr_reduction = 0.7 if validation_regression else 0.9
            cycle_reduction = 0.8 if validation_regression else 0.9

            logger.info(f"Applying partial penalty: reducing learning rate by {1-lr_reduction:.0%} and "
                       f"cycle length by {1-cycle_reduction:.0%} without backtracking.")

            self.current_max_lr *= lr_reduction

            # Calculate new cycle length with safety bounds
            new_cycle_length = int(self.current_cycle_length * cycle_reduction)
            new_cycle_length = max(self.min_cycle_length, new_cycle_length)

            # Check if cycle length is too small to continue
            if new_cycle_length <= 10:
                logger.info(f"Cycle length after partial penalty ({new_cycle_length}) is too small. Signaling to end training.")
                control.should_training_stop = True
                new_cycle_length = max(1, new_cycle_length)

            self.current_cycle_length = new_cycle_length
            self.metrics.reset()
            self._create_lr_scheduler()
            return  # End processing after applying partial penalty

        # Log cycle metrics to history
        self.metric_history['cycle_lengths'].append(self.current_cycle_length)
        self.metric_history['max_lr_per_cycle'].append(self.current_max_lr)
        self.metric_history['loss_volatility'].append(self.metrics.volatility)
        self.metric_history['gen_gaps'].append(self.metrics.gen_gap)
        self.metric_history['steps'].append(state.global_step)

        # Log metrics to wandb if available
        try:
            if wandb.run is not None:
                wandb.log({
                    "adaptive_lr/cycle": self.current_cycle,
                    "adaptive_lr/cycle_length": self.current_cycle_length,
                    "adaptive_lr/max_lr": self.current_max_lr,
                    "adaptive_lr/loss_volatility": self.metrics.volatility,
                    "adaptive_lr/overfit_ratio": overfit_ratio,
                    "adaptive_lr/cycles_without_improvement": self.cycles_without_improvement,
                    "adaptive_lr/validation_loss_regression": 1 if validation_regression else 0
                }, step=state.global_step)
        except Exception:
            pass

        # Compute next cycle length and next maximum LR
        next_cycle_length = self._compute_next_cycle_length(
            volatility=self.metrics.volatility,
            gen_gap=self.metrics.gen_gap,
            training_improvement=(self.cycles_without_improvement == 0)
        )

        # Handle small cycle length
        if next_cycle_length <= 10:
            logger.info(f"Computed next cycle length ({next_cycle_length}) is too small. Signaling to end training.")
            control.should_training_stop = True
            next_cycle_length = max(1, next_cycle_length)

        next_max_lr = self._compute_next_max_lr(
            volatility=self.metrics.volatility,
            gen_gap=self.metrics.gen_gap,
            training_improvement=(self.cycles_without_improvement == 0)
        )

        logger.info("===== CYCLE TRANSITION DETAILS =====")
        logger.info(f"Previous cycle length: {self.current_cycle_length}, Next cycle length: {next_cycle_length}")
        logger.info(f"Previous max LR: {self.current_max_lr:.6f}, Next max LR: {next_max_lr:.6f}")

        # Reset metrics and prepare for the new cycle
        self.metrics.reset()
        self.current_cycle_length = next_cycle_length
        self.current_max_lr = next_max_lr

        logger.info(f"Creating new LR scheduler for cycle {self.current_cycle}")
        try:
            self._create_lr_scheduler()
        except Exception as e:
            logger.error(f"Failed to create LR scheduler: {e}. Stopping training.")
            control.should_training_stop = True
            return

        # Reset the step counter for the new cycle
        self.step_in_cycle = 0

    def _compute_next_cycle_length(self, volatility: float, gen_gap: float, training_improvement: bool) -> int:
        """
        Compute the optimal length for the next training cycle based on observed metrics.

        Args:
            volatility: Measure of training loss volatility
            gen_gap: Generalization gap (val_loss - train_loss)
            training_improvement: Whether training showed improvement in this cycle

        Returns:
            int: Number of steps for the next cycle
        """
        # Start with base growth factor from current cycle length
        next_length = int(self.current_cycle_length * self.base_cycle_growth_factor)
        adjustment_reasons = []
        adjustment_reasons.append(f"base growth factor ({self.base_cycle_growth_factor:.2f})")

        # Apply adjustments based on training signals
        if training_improvement:
            # For successful cycles, we can increase the cycle length more aggressively
            next_length = int(next_length * self.adaptive_cycle_growth)
            adjustment_reasons.append("training improvement")

        # Check for overfitting signals
        if gen_gap < -self.generalization_threshold:
            reduction_factor = max(0.7, 1.0 - (abs(gen_gap) / (2 * self.generalization_threshold)))
            next_length = int(next_length * reduction_factor)
            adjustment_reasons.append(f"potential overfitting (gap={gen_gap:.4f}, factor={reduction_factor:.2f})")

        # Adjust based on loss volatility
        if volatility < 0.05:
            # Stable loss suggests we can use larger cycles
            next_length = int(next_length * 1.2)
            adjustment_reasons.append(f"stable loss (volatility={volatility:.4f})")
        elif volatility > 0.3:
            # High volatility suggests we should use shorter cycles
            volatility_factor = max(0.8, 1.0 - ((volatility - 0.3) * 0.5))
            next_length = int(next_length * volatility_factor)
            adjustment_reasons.append(f"volatile loss (volatility={volatility:.4f}, factor={volatility_factor:.2f})")

        # Adjust based on cycles without improvement
        if self.cycles_without_improvement > 0:
            # As we approach convergence, use longer cycles to fine-tune
            extend_factor = 1.0 + 0.2 * self.cycles_without_improvement
            next_length = int(next_length * extend_factor)
            adjustment_reasons.append(f"approaching convergence ({self.cycles_without_improvement} cycles w/o improvement)")

        # Apply bounds and log adjustments
        next_length = max(self.min_cycle_length, min(self.max_cycle_length, next_length))

        # Log adjustment reasons
        if adjustment_reasons:
            logger.info(f"Cycle length adjusted to {next_length} due to: {', '.join(adjustment_reasons)}")

        # Final safety check for small cycle lengths
        if next_length <= 10:
            logger.info(f"Computed cycle length is too small ({next_length}). Will signal to end training.")
            # Return a small but valid value (instead of 0) to avoid division by zero
            return max(1, next_length)

        return next_length

    def _compute_next_max_lr(self, volatility: float, gen_gap: float, training_improvement: bool) -> float:
        """
        Compute the optimal maximum learning rate for the next cycle based on observed metrics.

        Args:
            volatility: Measure of training loss volatility
            gen_gap: Generalization gap (val_loss - train_loss)
            training_improvement: Whether training showed improvement in this cycle

        Returns:
            float: Maximum learning rate for the next cycle
        """
        next_lr = self.current_max_lr
        adjustment_reasons = []

        # Positive generalization gap suggests underfitting - we can increase LR
        if gen_gap >= 0:
            # Calculate a quality score based on gap and stability
            quality_ratio = gen_gap / self.generalization_threshold
            stability_score = max(0, 1.0 - (volatility / 0.1))
            training_quality = quality_ratio * stability_score

            # Determine increase factor based on quality score
            if training_quality > 1.5:
                increase_factor = min(1.3, 1.0 + (0.15 * training_quality))
                adjustment_reasons.append(f"excellent training signals (quality={training_quality:.2f})")
            elif training_quality > 0.8:
                increase_factor = min(1.15, 1.0 + (0.08 * training_quality))
                adjustment_reasons.append(f"good training signals (quality={training_quality:.2f})")
            elif training_quality > 0.3:
                increase_factor = min(1.05, 1.0 + (0.03 * training_quality))
                adjustment_reasons.append(f"decent training signals (quality={training_quality:.2f})")
            else:
                increase_factor = 1.0
                adjustment_reasons.append(f"minimal positive signals (quality={training_quality:.2f})")

            # Consider available headroom for LR increases
            headroom_ratio = self.initial_lr / max(next_lr, 1e-10)
            if headroom_ratio > 1.01:
                headroom_factor = min(1.0, 0.3 + (0.7 * (headroom_ratio - 1.0) / headroom_ratio))
                effective_increase = 1.0 + ((increase_factor - 1.0) * headroom_factor)
                next_lr *= effective_increase
                adjustment_reasons.append(f"LR headroom factor: {headroom_factor:.2f}")

            # Consecutive good cycles bonus
            if training_improvement and self.current_cycle > 0:
                consecutive_bonus = min(1.05, 1.0 + (0.01 * self.current_cycle))
                next_lr *= consecutive_bonus
                adjustment_reasons.append(f"consecutive good cycles bonus: {consecutive_bonus:.2f}")

        # Negative generalization gap suggests overfitting - reduce LR
        if gen_gap < 0:
            severity_ratio = abs(gen_gap) / self.generalization_threshold
            if severity_ratio <= 1.0:
                reduction_factor = max(0.8, 1.0 - (severity_ratio * 0.2))
                adjustment_reasons.append(f"mild overfitting (gap={gen_gap:.4f}, severity={severity_ratio:.2f})")
            elif severity_ratio <= 2.0:
                reduction_factor = max(0.6, 0.8 - ((severity_ratio - 1.0) * 0.2))
                adjustment_reasons.append(f"moderate overfitting (gap={gen_gap:.4f}, severity={severity_ratio:.2f})")
            else:
                reduction_factor = max(0.3, 0.6 - min(0.3, ((severity_ratio - 2.0) * 0.1)))
                adjustment_reasons.append(f"severe overfitting (gap={gen_gap:.4f}, severity={severity_ratio:.2f})")

            next_lr *= reduction_factor

            # Early cycle overfit protection - more aggressive early
            if self.current_cycle <= 1:
                early_cycle_factor = 0.8
                next_lr *= early_cycle_factor
                adjustment_reasons.append(f"early-cycle overfitting protection")

        # Adjust for high volatility
        if volatility > 0.1:
            volatility_multiplier = max(0.5, 1.0 - (volatility * self.volatility_penalty))
            next_lr *= volatility_multiplier
            adjustment_reasons.append(f"high volatility ({volatility:.4f})")

        # Adjust for cycles without improvement
        if self.cycles_without_improvement > 0:
            reduction = 0.8 ** self.cycles_without_improvement
            next_lr *= reduction
            adjustment_reasons.append(f"cycles without improvement: {self.cycles_without_improvement} (reduction={reduction:.2f})")

        # Ensure LR stays within bounds
        next_lr = max(self.min_lr, min(self.initial_lr, next_lr))
        next_lr = max(1e-10, next_lr)

        # Log adjustment reasons
        if adjustment_reasons:
            logger.info(f"Learning rate adjusted to {next_lr:.2e} due to: {', '.join(adjustment_reasons)}")

        return next_lr

class GradientNoiseCallback(TrainerCallback):
    """
    Adds targeted noise to parameters that show signs of memorization (overfitting)
    by tracking differences between gradients on training and validation data.
    """
    def __init__(
        self,
        base_noise_scale: float = 0.01,
        max_noise_scale: float = 0.05,
        memory_window_size: int = 10,
        memorization_threshold: float = 1.5,
        validation_batch_frequency: int = 50,
        ema_decay: float = 0.9,
        log_frequency: int = 50
    ):
        self.base_noise_scale = base_noise_scale
        self.max_noise_scale = max_noise_scale
        self.memory_window_size = memory_window_size
        self.memorization_threshold = memorization_threshold
        self.validation_batch_frequency = validation_batch_frequency
        self.ema_decay = ema_decay
        self.log_frequency = log_frequency

        # Tracking variables
        self.param_train_grad_ema = {}  # EMA of training gradients
        self.param_val_grad_ema = {}    # EMA of validation gradients
        self.memorization_scores = {}   # Current memorization score per parameter
        self.step_count = 0
        self.val_dataloader = None
        self.val_iter = None
        self.last_logged_step = -1
        self.params_modified = 0
        self.total_params = 0

    def on_train_begin(self, args, state, control, **kwargs):
        """Initialize validation dataloader iterator"""
        if not hasattr(args, 'eval_dataset') or args.eval_dataset is None:
            logger.warning("MemorizationAwareNoiseCallback requires eval_dataset, but none was provided")
            return

        try:
            # Get validation dataloader from trainer
            self.val_dataloader = kwargs.get('train_dataloader').__class__(
                args.eval_dataset,
                batch_size=kwargs.get('train_dataloader').batch_size,
                collate_fn=kwargs.get('train_dataloader').collate_fn
            )
            self.val_iter = iter(self.val_dataloader)
            logger.info(f"MemorizationAwareNoiseCallback initialized with validation dataloader")
        except Exception as e:
            logger.error(f"Failed to initialize validation dataloader: {e}")

    def _get_validation_batch(self):
        """Get the next validation batch, reinitializing iterator if needed"""
        if self.val_dataloader is None:
            return None

        try:
            # Try to get next batch
            batch = next(self.val_iter)
        except StopIteration:
            # Reinitialize iterator and try again
            self.val_iter = iter(self.val_dataloader)
            try:
                batch = next(self.val_iter)
            except StopIteration:
                logger.error("Validation dataset appears to be empty")
                return None

        # Move batch to correct device if needed
        device = next(self.trainer.model.parameters()).device
        batch = {k: v.to(device) if hasattr(v, 'to') else v for k, v in batch.items()}

        return batch

    def on_step_begin(self, args, state, control, **kwargs):
        """Track step count for validation batch scheduling"""
        self.step_count += 1

    def _update_grad_ema(self, model, validation_batch=None):
        """
        Update gradient EMAs and memorization scores.
        If validation_batch is provided, update validation gradient EMA.
        Otherwise, update training gradient EMA (from current step).
        """
        is_validation = validation_batch is not None
        ema_dict = self.param_val_grad_ema if is_validation else self.param_train_grad_ema

        # For validation, perform forward/backward pass without affecting model
        if is_validation:
            # Store current model state
            optimizer_state = self.trainer.optimizer.state_dict()

            # Forward/backward pass on validation batch
            self.trainer.model.zero_grad()
            with torch.enable_grad():  # Ensure we get gradients
                outputs = self.trainer.model(**validation_batch)
                if isinstance(outputs, dict):
                    loss = outputs["loss"] if "loss" in outputs else outputs.loss
                else:
                    loss = outputs.loss
                loss.backward()

            # Restore optimizer state after validation grad computation
            self.trainer.optimizer.load_state_dict(optimizer_state)

        # Update EMA for each parameter's gradient
        for name, param in model.named_parameters():
            if param.grad is None:
                continue

            # Compute normalized absolute gradient
            grad_norm = torch.norm(param.grad.abs()).item() / torch.norm(param.data).item()

            # Initialize EMA for this parameter if needed
            if name not in ema_dict:
                ema_dict[name] = grad_norm
            else:
                ema_dict[name] = self.ema_decay * ema_dict[name] + (1 - self.ema_decay) * grad_norm

            # Update memorization score if we have both training and validation gradients
            if not is_validation and name in self.param_val_grad_ema and name in self.param_train_grad_ema:
                if self.param_val_grad_ema[name] > 1e-10:  # Avoid division by zero
                    # Memorization = ratio of training gradient to validation gradient
                    # Higher ratio = parameter learns more on training than validation = memorization
                    ratio = self.param_train_grad_ema[name] / self.param_val_grad_ema[name]

                    if name not in self.memorization_scores:
                        self.memorization_scores[name] = ratio
                    else:
                        # Smooth update of memorization score
                        self.memorization_scores[name] = 0.8 * self.memorization_scores[name] + 0.2 * ratio

    def on_pre_optimizer_step(self, args, state, control, model=None, optimizer=None, **kwargs):
        """
        Main logic: compute memorization scores and apply targeted noise
        """
        if model is None or optimizer is None or self.val_dataloader is None:
            return

        # Get current model
        model = model.module if hasattr(model, "module") else model

        # Periodically compute validation gradients
        if self.step_count % self.validation_batch_frequency == 0:
            validation_batch = self._get_validation_batch()
            if validation_batch is not None:
                self._update_grad_ema(model, validation_batch)

        # Always update training gradient EMA
        self._update_grad_ema(model)

        # Apply noise to parameters based on memorization scores
        with torch.no_grad():
            self.params_modified = 0
            self.total_params = 0

            for name, param in model.named_parameters():
                self.total_params += 1
                if param.grad is None or name not in self.memorization_scores:
                    continue

                # Get memorization score and determine if it's high enough for noise
                memo_score = self.memorization_scores[name]
                if memo_score > self.memorization_threshold:
                    # Scale noise based on memorization score
                    # More memorization = more noise
                    severity = min(3.0, (memo_score - self.memorization_threshold) / self.memorization_threshold)
                    noise_scale = min(
                        self.max_noise_scale,
                        self.base_noise_scale * (1.0 + severity)
                    )

                    # Scale by parameter norm for stability
                    param_norm = torch.norm(param.data)
                    scaled_noise = torch.randn_like(param.grad) * noise_scale * param_norm

                    # Add noise to gradient
                    param.grad.add_(scaled_noise)
                    self.params_modified += 1

        # Log progress periodically
        if self.step_count % self.log_frequency == 0 and self.step_count > self.last_logged_step:
            memo_params_pct = (self.params_modified / max(1, self.total_params)) * 100

            logger.info(f"Memorization-aware noise: Applied to {self.params_modified}/{self.total_params} "
                       f"parameters ({memo_params_pct:.1f}%)")

            # Log highest memorization scores for visibility
            if self.memorization_scores:
                top_memo = sorted(
                    [(k, v) for k, v in self.memorization_scores.items()],
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                logger.info(f"Top memo {top_memo}")

            # Try to log to wandb if available
            try:
                if wandb.run is not None:
                    wandb.log({
                        "grad_noise/memorization_params_pct": memo_params_pct,
                        "grad_noise/max_memorization_score": max(self.memorization_scores.values(), default=0),
                        "grad_noise/mean_memorization_score": sum(self.memorization_scores.values()) / max(1, len(self.memorization_scores)),
                    }, step=state.global_step)
            except Exception:
                pass

            self.last_logged_step = self.step_count


class ElementWiseOptimizerDropoutCallback(TrainerCallback):
    """
    Implements element-wise gradient dropout during training.

    This callback randomly zeros out individual weight gradients (not entire parameter tensors)
    with a specified probability, providing a fine-grained regularization technique.
    """

    def __init__(
        self,
        dropout_prob: float = 0.1,
        exclude_bias: bool = True,
        log_frequency: int = 50
    ):
        """
        Initialize the ElementWiseOptimizerDropoutCallback.

        Args:
            dropout_prob: Probability of dropping out each individual gradient element
            exclude_bias: Whether to exclude bias parameters from dropout
            log_frequency: How often to log dropout statistics
        """
        self.dropout_prob = dropout_prob
        self.exclude_bias = exclude_bias
        self.log_frequency = log_frequency

        # State tracking
        self.step_count = 0
        self.last_logged_step = -1

        logger.info(f"Initialized ElementWiseOptimizerDropoutCallback with dropout_prob={dropout_prob}")

    def on_step_begin(self, args, state, control, **kwargs):
        """Track step count for logging"""
        self.step_count += 1

    def on_pre_optimizer_step(self, args, state, control, model=None, optimizer=None, **kwargs):
        """Apply element-wise gradient dropout before optimizer step"""
        if model is None or optimizer is None:
            return

        # Get current model
        model = model.module if hasattr(model, "module") else model

        # Dropout statistics
        total_elements = 0
        elements_zeroed = 0
        params_affected = 0
        total_params = 0

        # Apply dropout to individual gradient elements
        with torch.no_grad():
            for name, param in model.named_parameters():
                # Skip parameters without gradients
                if param.grad is None:
                    continue

                # Skip bias parameters if configured
                if self.exclude_bias and "bias" in name:
                    continue

                total_params += 1

                # Create a binary dropout mask (same shape as gradient tensor)
                # 1 = keep gradient, 0 = zero gradient
                dropout_mask = torch.bernoulli(
                    torch.ones_like(param.grad) * (1 - self.dropout_prob)
                ).to(param.grad.device)

                # Count elements before applying mask
                param_total_elements = param.grad.numel()
                total_elements += param_total_elements

                # Count how many elements will be zeroed
                zeroed_count = param_total_elements - int(dropout_mask.sum().item())
                elements_zeroed += zeroed_count

                # Apply the mask (element-wise multiply)
                param.grad.mul_(dropout_mask)

                # Track if any elements in this param were affected
                if zeroed_count > 0:
                    params_affected += 1

        # Log progress periodically
        if self.step_count % self.log_frequency == 0 and self.step_count > self.last_logged_step:
            if total_elements > 0:
                elements_pct = (elements_zeroed / total_elements) * 100
                params_pct = (params_affected / total_params) * 100

                logger.info(f"Optimizer dropout: Zeroed {elements_zeroed:,}/{total_elements:,} "
                           f"gradient elements ({elements_pct:.2f}%) affecting {params_affected}/{total_params} "
                           f"parameters ({params_pct:.2f}%)")

                # Try to log to wandb if available
                try:
                    import wandb
                    if wandb.run is not None:
                        wandb.log({
                            "optimizer_dropout/elements_zeroed_pct": elements_pct,
                            "optimizer_dropout/params_affected_pct": params_pct,
                            "optimizer_dropout/elements_zeroed": elements_zeroed,
                        }, step=state.global_step)
                except (ImportError, Exception):
                    pass

            self.last_logged_step = self.step_count

class DynamicGradientAccumulationCallback(TrainerCallback):
    """
    Dynamically adjusts gradient accumulation steps based on the observed
    variance in gradients across batches.

    This callback monitors gradient statistics and automatically increases
    accumulation when gradients show high variance (less stable training),
    or decreases accumulation when variance is low (more stable training).

    Args:
        min_accumulation_steps (int, defaults to 1):
            Minimum allowed gradient accumulation steps
        max_accumulation_steps (int, defaults to 16):
            Maximum allowed gradient accumulation steps
        initial_accumulation_steps (Optional[int], defaults to None):
            Starting number of accumulation steps (if None, will use trainer's value)
        increase_threshold (float, defaults to 0.5):
            Variance threshold for increasing accumulation
        decrease_threshold (float, defaults to 0.2):
            Variance threshold for decreasing accumulation
        adjustment_window (int, defaults to 100):
            Steps between adjustment decisions
        adjustment_cooldown (int, defaults to 200):
            Minimum steps between actual adjustments
        increase_factor (float, defaults to 2.0):
            Multiply steps by this when increasing
        decrease_factor (float, defaults to 0.5):
            Multiply steps by this when decreasing
        ema_decay (float, defaults to 0.95):
            Decay rate for exponential moving average of gradient stats
        log_frequency (int, defaults to 10):
            How often to log statistics
        startup_delay_steps (int, defaults to 100):
            Wait this many steps before first adjustment
    """

    def __init__(
        self,
        # Required parameters
        trainer: Optional["Trainer"] = None,  # Reference to Trainer object

        # Accumulation bounds
        min_accumulation_steps: int = 1,
        max_accumulation_steps: int = 16,
        initial_accumulation_steps: Optional[int] = None,

        # Variance thresholds
        increase_threshold: float = 0.5,  # Increase accumulation if variance > this
        decrease_threshold: float = 0.2,  # Decrease accumulation if variance < this

        # Adjustment frequency
        adjustment_window: int = 100,  # Steps between adjustment decisions
        adjustment_cooldown: int = 200,  # Min steps between adjustments

        # Change magnitude
        increase_factor: float = 2.0,  # Multiply steps by this when increasing
        decrease_factor: float = 0.5,  # Multiply steps by this when decreasing

        # Tracking
        ema_decay: float = 0.95,  # Decay for exponential moving average
        log_frequency: int = 10,  # How often to log stats
        startup_delay_steps: int = 100,  # Wait this many steps before first adjustment
    ):
        """
        Initialize the dynamic gradient accumulation callback.

        Args:
            min_accumulation_steps: Minimum allowed gradient accumulation steps
            max_accumulation_steps: Maximum allowed gradient accumulation steps
            initial_accumulation_steps: Starting number of accumulation steps
                                        (if None, will use trainer's value)
            increase_threshold: Variance threshold for increasing accumulation
            decrease_threshold: Variance threshold for decreasing accumulation
            adjustment_window: Steps between adjustment decisions
            adjustment_cooldown: Minimum steps between actual adjustments
            increase_factor: Multiply steps by this when increasing
            decrease_factor: Multiply steps by this when decreasing
            ema_decay: Decay rate for exponential moving average of gradient stats
            log_frequency: How often to log statistics
            startup_delay_steps: Wait this many steps before first adjustment
        """
        # Store configuration
        self.min_accumulation_steps = max(1, min_accumulation_steps)
        self.max_accumulation_steps = max(self.min_accumulation_steps, max_accumulation_steps)
        self.initial_accumulation_steps = initial_accumulation_steps
        self.increase_threshold = increase_threshold
        self.decrease_threshold = decrease_threshold
        self.adjustment_window = max(1, adjustment_window)
        self.adjustment_cooldown = max(0, adjustment_cooldown)
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self.ema_decay = ema_decay
        self.log_frequency = log_frequency
        self.startup_delay_steps = startup_delay_steps

        # Runtime state
        self.trainer = trainer  # Store trainer if provided during initialization
        self.last_adjustment_step: int = -float('inf')
        self.current_adjustment_window_start: int = 0
        self.step_count: int = 0
        self.gradient_stats_ema: Optional[Dict[str, float]] = None
        self.batch_gradient_norms: List[float] = []
        self.recent_variances: List[float] = []
        self.last_logged_step: int = -1

        # Track adjusted accumulation steps
        self.original_accumulation_steps: Optional[int] = None
        self.current_accumulation_steps: Optional[int] = None

        # Safety flags
        self.is_initialized: bool = False
        self.is_in_accumulation: bool = False
        self.skip_next_adjustment: bool = False
        self.seen_parameters: Set[str] = set()

        # New flags for optimization
        self.pending_adjustments = []  # Queue of pending adjustments
        self.is_optimizer_step = False  # Flag to detect optimizer steps
        self.last_global_step = -1     # Track global step changes

        logger.info(
            f"Initialized DynamicGradientAccumulationCallback with "
            f"min_steps={self.min_accumulation_steps}, "
            f"max_steps={self.max_accumulation_steps}"
        )

    def on_train_begin(self, args: TrainingArguments, state: TrainerState,
                       control: TrainerControl, **kwargs):
        """Initialize the callback when training begins."""
        # Only set trainer if it wasn't already set during initialization
        if self.trainer is None:
            trainer = kwargs.get("trainer", None)
            if trainer is None:
                logger.warning(
                    "Trainer instance not found in kwargs. "
                    "Dynamic gradient accumulation may not work correctly."
                )
                return
            self.trainer = trainer

        # Store original accumulation steps value
        self.original_accumulation_steps = args.gradient_accumulation_steps
        logger.info(f"Original gradient_accumulation_steps: {self.original_accumulation_steps}")

        # Set initial accumulation steps
        if self.initial_accumulation_steps is not None:
            # Clamp to valid range
            init_steps = max(self.min_accumulation_steps,
                           min(self.max_accumulation_steps, self.initial_accumulation_steps))

            if init_steps != args.gradient_accumulation_steps:
                logger.info(f"Setting initial gradient_accumulation_steps to {init_steps}")
                self._update_accumulation_steps(init_steps, args)

        # Store current value
        self.current_accumulation_steps = args.gradient_accumulation_steps

        # Reset counters and statistics
        self.step_count = 0
        self.batch_gradient_norms = []
        self.recent_variances = []
        self.gradient_stats_ema = None
        self.last_adjustment_step = -float('inf')
        self.current_adjustment_window_start = 0
        self.is_initialized = True

    def on_step_begin(self, args: TrainingArguments, state: TrainerState,
                     control: TrainerControl, **kwargs):
        """
        Track step count and monitor accumulation state.
        This also now handles gradient analysis that previously was in on_pre_optimizer_step.
        """
        if not self.is_initialized:
            return

        self.step_count += 1

        # Detect if this is an optimizer step by checking if global_step changed
        current_global_step = state.global_step
        self.is_optimizer_step = (current_global_step != self.last_global_step)
        self.last_global_step = current_global_step

        # Check if we're in the middle of accumulation
        # In Transformers Trainer, there are several ways to detect this:

        # 1. Check if trainer has steps_in_epoch attribute (newer versions)
        if hasattr(self.trainer, "steps_in_epoch"):
            steps_in_epoch = self.trainer.steps_in_epoch
            # If steps_in_epoch is an integer, we've completed a full accumulation step
            # and are about to start a new one
            is_start_of_accumulation = steps_in_epoch.is_integer()

            # Update accumulation state
            if is_start_of_accumulation and self.is_in_accumulation:
                self.is_in_accumulation = False

            if is_start_of_accumulation and not self.is_in_accumulation:
                self.is_in_accumulation = True

        # 2. Fallback detection based on global_step and gradient_accumulation_steps
        else:
            # In Trainer, optimizer step happens when (step + 1) % gradient_accumulation_steps == 0
            # So we're accumulating gradients when this is not the case
            self.is_in_accumulation = ((state.global_step + 1) % args.gradient_accumulation_steps != 0)

        # If this is the last step in an accumulation cycle, analyze gradients
        # This effectively replaces the on_pre_optimizer_step hook
        if self.is_optimizer_step and hasattr(self.trainer, "model"):
            model = self.trainer.model
            try:
                # Collect gradient statistics
                current_stats = self._collect_gradient_stats(model)

                if current_stats:
                    # Update EMA of stats
                    self._update_gradient_stats_ema(current_stats)

                    # Calculate variance of batch gradient norms
                    current_variance = self._calculate_gradient_variance()

                    # Log statistics periodically
                    if (
                        self.step_count % self.log_frequency == 0 and
                        self.step_count > self.last_logged_step
                    ):
                        self._log_gradient_stats(current_stats, current_variance, state.global_step)
                        self.last_logged_step = self.step_count

                    # Check if it's time to consider an adjustment
                    is_adjustment_window = (
                        self.step_count >= self.startup_delay_steps and
                        self.step_count - self.current_adjustment_window_start >= self.adjustment_window and
                        self.step_count - self.last_adjustment_step >= self.adjustment_cooldown and
                        not self.skip_next_adjustment
                    )

                    if is_adjustment_window and current_variance is not None:
                        # Decide whether and how to adjust
                        should_adjust, new_steps = self._decide_adjustment(current_variance)

                        if should_adjust:
                            # Update accumulation steps - now using queue-based approach
                            self.pending_adjustments.append(new_steps)
                            logger.info(f"Queuing adjustment to {new_steps} steps for next appropriate time")

                            self.last_adjustment_step = self.step_count
                            self.current_accumulation_steps = new_steps
                            self.current_adjustment_window_start = self.step_count
                        else:
                            # Reset adjustment window start if no adjustment needed
                            self.current_adjustment_window_start = self.step_count

                    # Reset the skip flag
                    self.skip_next_adjustment = False

            except Exception as e:
                logger.error(f"Error in dynamic gradient accumulation: {e}")
                import traceback
                logger.error(traceback.format_exc())

        # Apply any pending adjustments at the start of a new accumulation cycle
        if not self.is_in_accumulation and self.pending_adjustments:
            new_steps = self.pending_adjustments.pop(0)
            success = self._update_accumulation_steps(new_steps, args)
            if success:
                self.current_accumulation_steps = new_steps
                logger.info(f"Applied pending adjustment: gradient_accumulation_steps now {new_steps}")

    def _collect_gradient_stats(self, model: torch.nn.Module) -> Dict[str, float]:
        """
        Collect statistics about the current gradients.

        Args:
            model: The model with gradients to analyze

        Returns:
            Dictionary with gradient statistics
        """
        # Safety check
        if model is None:
            return {}

        # Get the model module (handle distributed training)
        model_module = model.module if hasattr(model, "module") else model

        # Calculate statistics
        total_norm = 0.0
        param_count = 0
        param_with_grad_count = 0
        grad_l2_norms = []

        for name, param in model_module.named_parameters():
            if param.requires_grad:
                param_count += 1
                self.seen_parameters.add(name)

                if param.grad is not None:
                    param_with_grad_count += 1
                    param_norm = param.grad.norm(2).item()
                    grad_l2_norms.append(param_norm)
                    total_norm += param_norm ** 2

        # Calculate batch gradient norm
        batch_grad_norm = math.sqrt(total_norm) if total_norm > 0 else 0.0

        # Store for variance calculation
        if batch_grad_norm > 0:
            self.batch_gradient_norms.append(batch_grad_norm)

            # Limit the size of the buffer
            if len(self.batch_gradient_norms) > 2 * self.adjustment_window:
                self.batch_gradient_norms = self.batch_gradient_norms[-self.adjustment_window:]

        # Compute descriptive statistics of parameter-wise gradient norms
        stats = {}

        if grad_l2_norms:
            stats["batch_grad_norm"] = batch_grad_norm
            stats["mean_param_grad_norm"] = np.mean(grad_l2_norms)
            stats["median_param_grad_norm"] = np.median(grad_l2_norms)
            stats["max_param_grad_norm"] = max(grad_l2_norms)
            stats["min_param_grad_norm"] = min(grad_l2_norms)
            stats["param_grad_norm_std"] = np.std(grad_l2_norms)

            # Coefficient of variation = std / mean (normalized measure of dispersion)
            if stats["mean_param_grad_norm"] > 0:
                stats["param_grad_norm_cv"] = stats["param_grad_norm_std"] / stats["mean_param_grad_norm"]
            else:
                stats["param_grad_norm_cv"] = 0.0

        stats["param_count"] = param_count
        stats["param_with_grad_count"] = param_with_grad_count
        stats["param_with_grad_pct"] = (param_with_grad_count / max(1, param_count)) * 100

        return stats

    def _update_gradient_stats_ema(self, current_stats: Dict[str, float]):
        """
        Update the exponential moving average of gradient statistics.

        Args:
            current_stats: Current batch gradient statistics
        """
        if self.gradient_stats_ema is None:
            # Initialize with current stats
            self.gradient_stats_ema = current_stats.copy()
        else:
            # Update EMA for each statistic
            for key, value in current_stats.items():
                if key in self.gradient_stats_ema:
                    self.gradient_stats_ema[key] = (
                        self.ema_decay * self.gradient_stats_ema[key] +
                        (1 - self.ema_decay) * value
                    )
                else:
                    self.gradient_stats_ema[key] = value

    def _calculate_gradient_variance(self) -> Optional[float]:
        """
        Calculate the variance of batch gradient norms over recent batches.

        Returns:
            Normalized variance or None if insufficient data
        """
        # We need at least 2 batches to compute variance
        if len(self.batch_gradient_norms) < 2:
            return None

        # Get the most recent window of batch norms
        window_size = min(len(self.batch_gradient_norms), self.adjustment_window)
        recent_norms = self.batch_gradient_norms[-window_size:]

        # Calculate variance and normalize by mean to get coefficient of variation
        mean_norm = np.mean(recent_norms)

        if mean_norm > 0:
            variance = np.var(recent_norms)
            normalized_variance = variance / (mean_norm ** 2)  # Squared coefficient of variation

            # Store for tracking
            self.recent_variances.append(normalized_variance)
            if len(self.recent_variances) > 10:
                self.recent_variances = self.recent_variances[-10:]

            return normalized_variance
        else:
            return 0.0

    def _update_accumulation_steps(self, new_steps: int, args: TrainingArguments):
        """
        Update the gradient accumulation steps in the trainer and arguments.

        Args:
            new_steps: New number of gradient accumulation steps
            args: TrainingArguments object to update

        Returns:
            bool: Whether update was successful
        """
        # Safety checks
        if new_steps == args.gradient_accumulation_steps:
            return True  # No change needed

        if new_steps < 1:
            logger.warning(f"Invalid accumulation steps: {new_steps}. Must be >= 1.")
            return False

        if self.is_in_accumulation:
            logger.warning(
                "Cannot change accumulation steps in the middle of accumulation. "
                "Skipping adjustment."
            )
            self.skip_next_adjustment = True
            return False

        try:
            # Update training arguments
            # Note: gradient_accumulation_steps is a property in TrainingArguments
            # that's backed by a private attribute _gradient_accumulation_steps
            old_steps = args.gradient_accumulation_steps
            args._gradient_accumulation_steps = new_steps

            # Check that the update was successful
            if args.gradient_accumulation_steps != new_steps:
                logger.warning(
                    f"Failed to update args.gradient_accumulation_steps to {new_steps}, "
                    f"it's still {args.gradient_accumulation_steps}"
                )
                return False

            logger.info(f"Updated gradient_accumulation_steps from {old_steps} to {new_steps}")
            return True
        except Exception as e:
            logger.error(f"Failed to update gradient_accumulation_steps: {e}")
            return False

    def _decide_adjustment(self, current_variance: float) -> Tuple[bool, int]:
        """
        Decide whether and how to adjust accumulation steps.

        Args:
            current_variance: Current normalized gradient variance

        Returns:
            Tuple of (should_adjust, new_steps)
        """
        current_steps = self.current_accumulation_steps

        # Apply adjustments based on thresholds
        if current_variance > self.increase_threshold:
            # High variance - increase accumulation steps
            new_steps = min(
                self.max_accumulation_steps,
                max(
                    current_steps + 1,
                    int(current_steps * self.increase_factor)
                )
            )
            reason = f"high variance ({current_variance:.4f} > {self.increase_threshold:.4f})"

        elif current_variance < self.decrease_threshold:
            # Low variance - decrease accumulation steps
            new_steps = max(
                self.min_accumulation_steps,
                min(
                    current_steps - 1,
                    int(current_steps * self.decrease_factor)
                )
            )
            reason = f"low variance ({current_variance:.4f} < {self.decrease_threshold:.4f})"

        else:
            # Variance is in the acceptable range
            return False, current_steps

        # Only adjust if the value will actually change
        if new_steps != current_steps:
            logger.info(
                f"Step {self.step_count}: Adjustment decision: {current_steps} -> {new_steps} due to {reason}"
            )
            return True, new_steps
        else:
            return False, current_steps

    def _log_gradient_stats(self, stats: Dict[str, float], variance: Optional[float], global_step: int):
        """
        Log gradient statistics for monitoring.

        Args:
            stats: Current gradient statistics
            variance: Current normalized gradient variance
            global_step: Global step for logging
        """
        # Only log if we have data to show
        if not stats:
            return

        # Prepare log message with key stats
        log_items = [
            f"Step {self.step_count}",
            f"Accumulation steps: {self.current_accumulation_steps}"
        ]

        # Add gradient norm if available
        if "batch_grad_norm" in stats:
            log_items.append(f"Batch grad norm: {stats['batch_grad_norm']:.4f}")

        # Add coefficient of variation if available
        if "param_grad_norm_cv" in stats:
            log_items.append(f"Param grad CV: {stats['param_grad_norm_cv']:.4f}")

        # Add variance if available
        if variance is not None:
            log_items.append(f"Batch norm variance: {variance:.4f}")
            mean_variance = np.mean(self.recent_variances) if self.recent_variances else 0
            log_items.append(f"Mean variance: {mean_variance:.4f}")

        logger.info(" | ".join(log_items))

        # Log to wandb if available
        try:
            if wandb.run is not None:
                log_dict = {
                    "dynamic_accum/grad_accum_steps": self.current_accumulation_steps,
                    "dynamic_accum/batch_grad_norm": stats.get("batch_grad_norm", 0),
                    "dynamic_accum/param_grad_cv": stats.get("param_grad_norm_cv", 0),
                }

                if variance is not None:
                    log_dict["dynamic_accum/batch_norm_variance"] = variance

                if self.recent_variances:
                    log_dict["dynamic_accum/mean_variance"] = np.mean(self.recent_variances)

                wandb.log(log_dict, step=global_step)
        except Exception:
            pass

    def on_train_end(self, args: TrainingArguments, state: TrainerState,
                    control: TrainerControl, **kwargs):
        """Restore original accumulation steps at the end of training."""
        if not self.is_initialized:
            return

        # Only restore if we modified the value
        if (
            self.original_accumulation_steps is not None and
            self.original_accumulation_steps != args.gradient_accumulation_steps
        ):
            logger.info(
                f"Restoring original gradient_accumulation_steps: {self.original_accumulation_steps}"
            )
            self._update_accumulation_steps(self.original_accumulation_steps, args)

        # Log seen parameters count
        logger.info(f"Total parameters tracked during training: {len(self.seen_parameters)}")

        # Cleanup
        self.batch_gradient_norms = []
        self.recent_variances = []
        self.gradient_stats_ema = None
        self.is_initialized = False

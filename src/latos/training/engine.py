"""Single-device float32 AdamW updates and token-weighted, read-only evaluation."""

import math
import time

import torch

from latos.doctor import available_backends, select_device
from latos.lora import LoRAModel
from latos.model.network import LatoModel
from latos.training.config import TrainingConfig
from latos.training.data import ShuffleStream, TokenDataset, collate


def check_dataset(model: LatoModel, dataset: TokenDataset) -> None:
    if (
        dataset.vocab_size != model.config.vocab_size
        or dataset.tokenizer_sha256 != model.config.tokenizer_sha256
        or dataset.sequence_length > model.config.context_length
    ):
        raise ValueError("Dataset does not match model vocabulary, tokenizer, or context")


def evaluate(model: LatoModel, dataset: TokenDataset, batch_size: int = 2) -> dict:
    """Score every target once, with no optimizer, RNG, or gradient changes."""
    check_dataset(model, dataset)
    if type(batch_size) is not int or not 1 <= batch_size <= 256:
        raise ValueError("Invalid validation batch size")
    device = str(model.embedding.weight.device)
    modes = [(module, module.training) for module in model.modules()]
    total_loss, targets = 0.0, 0
    try:
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(dataset.windows), batch_size):
                ids, labels, count = collate(
                    dataset,
                    list(range(start, min(start + batch_size, len(dataset.windows)))),
                    device,
                )
                loss = model(ids, labels).loss.item()
                if not math.isfinite(loss):
                    raise ValueError("Validation loss is nonfinite")
                total_loss += loss * count
                targets += count
    finally:
        for module, mode in modes:
            module.training = mode
    loss = total_loss / targets
    return {
        "loss": loss,
        "perplexity": math.exp(loss) if loss < 700 else None,
        "targets": targets,
        "windows": len(dataset.windows),
        "split": dataset.split,
    }


class Trainer:
    """Checkpoints are permitted only between completed optimizer updates.

    This model has no stochastic layers. The isolated shuffle generator is the
    only RNG consumed by training; global Python/NumPy/device RNGs are untouched.
    """

    def __init__(
        self,
        model: LatoModel,
        config: TrainingConfig,
        train: TokenDataset,
        validation: TokenDataset,
        device: str = "cpu",
    ):
        check_dataset(model, train)
        check_dataset(model, validation)
        if (train.target_masks is None) != (validation.target_masks is None):
            raise ValueError("Train and validation objectives must match")
        if train.split != "train" or validation.split != "validation":
            raise ValueError("Trainer requires separate train and validation splits")
        if (
            train.sequence_length != config.sequence_length
            or validation.sequence_length != config.sequence_length
        ):
            raise ValueError("Dataset sequence length differs from training configuration")
        if isinstance(model, LoRAModel):
            model.validate_adapter()
        elif any(p.dtype != torch.float32 or not p.requires_grad for p in model.parameters()):
            raise ValueError("Training requires float32, trainable model parameters")
        self.device = select_device(device, available_backends())
        self.model = model.to(self.device)
        self.config = config
        self.train_data = train
        self.validation_data = validation
        self.data_identities = {"train": train.identity, "validation": validation.identity}
        decay, no_decay = [], []
        for parameter in model.parameters():
            if not parameter.requires_grad:
                continue
            (decay if parameter.ndim >= 2 else no_decay).append(parameter)
        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": config.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
            foreach=False,
            fused=False,
        )
        self.optimizer.zero_grad(set_to_none=True)
        self.stream = ShuffleStream(len(train.windows), config.seed)
        self.step = 0
        self.tokens_seen = 0
        self.windows_seen = 0
        self.ready = True

    def update(self) -> dict:
        if not self.ready:
            raise ValueError("Trainer update failed or is in progress; reload the last checkpoint")
        if self.step >= self.config.max_steps:
            raise ValueError("Configured training run is complete")
        self.ready = False
        started = time.perf_counter()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        batches = [
            self.stream.take(self.config.batch_size) for _ in range(self.config.accumulation_steps)
        ]
        total = sum(self.train_data.target_count(i) for batch in batches for i in batch)
        loss_sum = 0.0
        for indices in batches:
            ids, labels, count = collate(self.train_data, indices, self.device)
            loss = self.model(ids, labels).loss
            if not torch.isfinite(loss).item():
                raise ValueError("Training loss is nonfinite; reload the last checkpoint")
            (loss * (count / total)).backward()
            loss_sum += loss.item() * count
        norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm,
            error_if_nonfinite=True,
            foreach=False,
        ).item()
        lr = self.config.learning_rate_at(self.step + 1)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.step()
        if any(not torch.isfinite(p).all().item() for p in self.model.parameters()):
            raise ValueError("Optimizer produced nonfinite weights; reload the last checkpoint")
        self.optimizer.zero_grad(set_to_none=True)
        self.step += 1
        self.tokens_seen += total
        self.windows_seen += sum(map(len, batches))
        self.ready = True
        elapsed = time.perf_counter() - started
        return {
            "step": self.step,
            "loss": loss_sum / total,
            "learning_rate": lr,
            "grad_norm_before_clip": norm,
            "targets": total,
            "tokens_seen": self.tokens_seen,
            "windows_seen": self.windows_seen,
            "epoch": self.stream.epoch,
            "seconds": elapsed,
            "targets_per_second": total / elapsed,
        }

    def validate(self) -> dict:
        return evaluate(self.model, self.validation_data, self.config.batch_size)

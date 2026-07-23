from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import random
import re
import statistics
import time
from typing import Any, Iterable

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from .model import ByteTokenizer, CARSRConfig, CARSRModel, load_checkpoint, save_checkpoint


THOMPSON_STATE_VERSION = "0.1.0"


@dataclass
class NormalInverseGammaPosterior:
    """Conjugate posterior for an arm with an unknown Gaussian reward mean/variance."""

    prior_mean: float = 0.0
    prior_strength: float = 1.0
    prior_alpha: float = 2.0
    prior_beta: float = 1.0
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def __post_init__(self) -> None:
        if self.prior_strength <= 0:
            raise ValueError("prior_strength must be positive")
        if self.prior_alpha <= 1:
            raise ValueError("prior_alpha must exceed one so the prior variance has a mean")
        if self.prior_beta <= 0:
            raise ValueError("prior_beta must be positive")
        if self.count < 0 or self.m2 < 0:
            raise ValueError("posterior sufficient statistics are invalid")

    def update(self, reward: float) -> None:
        if not math.isfinite(reward):
            raise ValueError("Thompson reward must be finite")
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (reward - self.mean)

    @property
    def parameters(self) -> tuple[float, float, float, float]:
        strength = self.prior_strength + self.count
        location = (
            self.prior_strength * self.prior_mean + self.count * self.mean
        ) / strength
        alpha = self.prior_alpha + 0.5 * self.count
        beta = self.prior_beta + 0.5 * self.m2
        if self.count:
            beta += (
                self.prior_strength
                * self.count
                * (self.mean - self.prior_mean) ** 2
                / (2.0 * strength)
            )
        return location, strength, alpha, beta

    @property
    def posterior_mean(self) -> float:
        return self.parameters[0]

    @property
    def posterior_std(self) -> float:
        _, strength, alpha, beta = self.parameters
        return math.sqrt(beta / ((alpha - 1.0) * strength))

    def sample(self, *, seed: int) -> float:
        location, strength, alpha, beta = self.parameters
        generator = random.Random(seed)
        precision = generator.gammavariate(alpha, 1.0 / beta)
        variance = 1.0 / max(precision, 1e-12)
        return generator.gauss(location, math.sqrt(variance / strength))

    def to_dict(self) -> dict[str, float | int]:
        return {
            "prior_mean": self.prior_mean,
            "prior_strength": self.prior_strength,
            "prior_alpha": self.prior_alpha,
            "prior_beta": self.prior_beta,
            "count": self.count,
            "mean": self.mean,
            "m2": self.m2,
            "posterior_mean": self.posterior_mean,
            "posterior_std": self.posterior_std,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "NormalInverseGammaPosterior":
        return cls(
            prior_mean=float(values["prior_mean"]),
            prior_strength=float(values["prior_strength"]),
            prior_alpha=float(values["prior_alpha"]),
            prior_beta=float(values["prior_beta"]),
            count=int(values.get("count", 0)),
            mean=float(values.get("mean", 0.0)),
            m2=float(values.get("m2", 0.0)),
        )


@dataclass
class ThompsonArm:
    name: str
    overrides: dict[str, Any]
    posterior: NormalInverseGammaPosterior
    pulls: int = 0
    allocated_steps: int = 0
    last_reward: float | None = None
    last_summary: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "overrides": self.overrides,
            "posterior": self.posterior.to_dict(),
            "pulls": self.pulls,
            "allocated_steps": self.allocated_steps,
            "last_reward": self.last_reward,
            "last_summary": self.last_summary,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ThompsonArm":
        return cls(
            name=str(values["name"]),
            overrides=dict(values["overrides"]),
            posterior=NormalInverseGammaPosterior.from_dict(values["posterior"]),
            pulls=int(values.get("pulls", 0)),
            allocated_steps=int(values.get("allocated_steps", 0)),
            last_reward=(
                None if values.get("last_reward") is None else float(values["last_reward"])
            ),
            last_summary={
                str(key): float(value)
                for key, value in dict(values.get("last_summary", {})).items()
            },
        )


class ThompsonResearchScheduler:
    """Resumable outer-loop allocator for expensive CARS-R research arms."""

    def __init__(
        self,
        arms: list[ThompsonArm],
        *,
        seed: int,
        min_pulls: int = 1,
        round_index: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not arms:
            raise ValueError("at least one Thompson arm is required")
        if min_pulls < 0:
            raise ValueError("min_pulls cannot be negative")
        names = [arm.name for arm in arms]
        if len(names) != len(set(names)):
            raise ValueError("Thompson arm names must be unique")
        self.arms = {arm.name: arm for arm in arms}
        self.seed = seed
        self.min_pulls = min_pulls
        self.round_index = round_index
        self.metadata = dict(metadata or {})

    def select(self) -> tuple[ThompsonArm, dict[str, float | None]]:
        ordered = [self.arms[name] for name in sorted(self.arms)]
        pending = [arm for arm in ordered if arm.pulls < self.min_pulls]
        if pending:
            minimum = min(arm.pulls for arm in pending)
            candidates = [arm for arm in pending if arm.pulls == minimum]
            selected = candidates[self.round_index % len(candidates)]
            samples = {arm.name: None for arm in ordered}
            return selected, samples

        samples = {
            arm.name: arm.posterior.sample(
                seed=self.seed + self.round_index * 104729 + index * 1009
            )
            for index, arm in enumerate(ordered)
        }
        selected = max(ordered, key=lambda arm: (samples[arm.name], arm.name))
        return selected, samples

    def update(
        self,
        name: str,
        *,
        reward: float,
        allocated_steps: int,
        summary: dict[str, float],
    ) -> None:
        arm = self.arms[name]
        arm.posterior.update(reward)
        arm.pulls += 1
        arm.allocated_steps += allocated_steps
        arm.last_reward = reward
        arm.last_summary = dict(summary)
        self.round_index += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": THOMPSON_STATE_VERSION,
            "seed": self.seed,
            "min_pulls": self.min_pulls,
            "round_index": self.round_index,
            "metadata": self.metadata,
            "arms": [self.arms[name].to_dict() for name in sorted(self.arms)],
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ThompsonResearchScheduler":
        if values.get("version") != THOMPSON_STATE_VERSION:
            raise ValueError("Thompson scheduler state is incompatible with CARS-R 0.1.0")
        return cls(
            [ThompsonArm.from_dict(item) for item in values["arms"]],
            seed=int(values["seed"]),
            min_pulls=int(values["min_pulls"]),
            round_index=int(values.get("round_index", 0)),
            metadata=dict(values.get("metadata", {})),
        )

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(target)

    @classmethod
    def load(cls, path: str | Path) -> "ThompsonResearchScheduler":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


DEMO_TEXT = """
CARS-R studies a narrow question: can a byte-level language model learn where to
compress bytes into causal patches and reuse a shared patch block for controllable
latent depth? Exact byte states remain in the local path. Global computation runs
on patches. The architecture is evaluated against dense and fixed-patch controls
at matched data, optimization steps, parameters, and recurrent depth.

Research claims require controlled evidence. Report next-byte loss, bits per byte,
exact digits and operators, patch lengths, router gradients, training throughput,
prefill latency, decoding latency, cache memory, and quality as recurrent depth
changes. Negative results and router collapse are results, not inconveniences.
""" * 64


class ByteWindowDataset(Dataset[Tensor]):
    def __init__(self, tokens: Tensor, sequence_length: int, samples: int, seed: int) -> None:
        if tokens.ndim != 1 or tokens.numel() < sequence_length + 1:
            raise ValueError("corpus must be longer than sequence_length")
        self.tokens = tokens
        self.sequence_length = sequence_length
        self.samples = samples
        generator = random.Random(seed)
        limit = tokens.numel() - sequence_length - 1
        self.starts = [generator.randrange(limit + 1) for _ in range(samples)]

    def __len__(self) -> int:
        return self.samples

    def __getitem__(self, index: int) -> Tensor:
        start = self.starts[index]
        return self.tokens[start : start + self.sequence_length + 1]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_corpus(path: str | None) -> Tensor:
    text = DEMO_TEXT if path is None else Path(path).read_text(encoding="utf-8")
    tokenizer = ByteTokenizer()
    return torch.tensor(tokenizer.encode(text, add_eos=True), dtype=torch.long)


def build_config(args: argparse.Namespace, *, patch_mode: str | None = None) -> CARSRConfig:
    config = CARSRConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_kv_heads=args.n_kv_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        byte_layers=args.byte_layers,
        byte_window=args.byte_window,
        patch_layers=args.patch_layers,
        decoder_layers=args.decoder_layers,
        max_seq_len=args.sequence_length,
        patch_mode=patch_mode or args.patch_mode,
        patch_attention=args.patch_attention,
        target_patch_ratio=args.patch_ratio,
        min_patch_size=args.min_patch_size,
        max_patch_size=args.max_patch_size,
        compression_loss_weight=args.compression_weight,
        cpla_content_dim=args.cpla_content_dim,
        cpla_position_dim=args.cpla_position_dim,
        span_mass_bias=not args.disable_span_mass_bias,
        recurrent_depths=tuple(sorted(set(args.depths))),
        default_recurrent_depth=args.default_depth,
        cache_dtype=args.cache_dtype,
    )
    config.validate()
    return config


def make_loader(
    tokens: Tensor,
    *,
    sequence_length: int,
    batch_size: int,
    steps: int,
    seed: int,
    shuffle: bool,
) -> DataLoader[Tensor]:
    samples = max(batch_size * steps, batch_size)
    dataset = ByteWindowDataset(tokens, sequence_length, samples, seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)


def exact_category_masks(labels: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    digits = (labels >= ord("0")) & (labels <= ord("9"))
    symbol_ids = torch.tensor(
        list(b"+-*/%=<>[]{}()&|^~.,:;_$#@!?'\\\""),
        device=labels.device,
    )
    symbols = (labels.unsqueeze(-1) == symbol_ids).any(-1)
    return digits, symbols, digits | symbols


@torch.inference_mode()
def evaluate(
    model: CARSRModel,
    loader: Iterable[Tensor],
    *,
    device: torch.device,
    depths: tuple[int, ...],
    max_batches: int,
) -> dict[str, float]:
    model.eval()
    result: dict[str, float] = {}
    for depth in depths:
        loss_sum = token_count = correct = 0.0
        digit_total = digit_correct = symbol_total = symbol_correct = 0.0
        risk_total = risk_correct = 0.0
        patch_length_sum = patch_length_square_sum = patch_total = 0.0
        patch_length_min = math.inf
        patch_length_max = 0.0
        boundary_sum = boundary_count = 0.0
        recurrence_norm_sum = recurrence_norm_count = 0.0
        batches = 0
        for batch in loader:
            batch = batch.to(device)
            inputs, labels = batch[:, :-1], batch[:, 1:]
            output = model(inputs, labels=labels, recurrent_depth=depth)
            assert output.token_loss is not None
            valid = labels.ne(model.config.pad_token_id)
            predictions = output.logits.argmax(-1)
            digits, symbols, risk = exact_category_masks(labels)
            digits &= valid
            symbols &= valid
            risk &= valid
            tokens = valid.sum().item()
            loss_sum += float(output.token_loss) * tokens
            token_count += tokens
            correct += ((predictions == labels) & valid).sum().item()
            digit_total += digits.sum().item()
            digit_correct += ((predictions == labels) & digits).sum().item()
            symbol_total += symbols.sum().item()
            symbol_correct += ((predictions == labels) & symbols).sum().item()
            risk_total += risk.sum().item()
            risk_correct += ((predictions == labels) & risk).sum().item()

            lengths = output.patching.patch_lengths[output.patching.patch_mask].float()
            if lengths.numel():
                patch_length_sum += lengths.sum().item()
                patch_length_square_sum += lengths.square().sum().item()
                patch_total += lengths.numel()
                patch_length_min = min(patch_length_min, lengths.min().item())
                patch_length_max = max(patch_length_max, lengths.max().item())
            signal = output.patching.boundary_probs[valid]
            boundary_sum += signal.sum().item()
            boundary_count += signal.numel()
            recurrence_norm_sum += output.recurrence_update_norms.sum().item()
            recurrence_norm_count += output.recurrence_update_norms.numel()
            batches += 1
            if batches >= max_batches:
                break

        mean_loss = loss_sum / max(token_count, 1.0)
        mean_patch = patch_length_sum / max(patch_total, 1.0)
        patch_variance = max(
            patch_length_square_sum / max(patch_total, 1.0) - mean_patch**2,
            0.0,
        )
        prefix = f"depth_{depth}"
        result.update(
            {
                f"{prefix}/loss": mean_loss,
                f"{prefix}/bits_per_byte": mean_loss / math.log(2.0),
                f"{prefix}/accuracy": correct / max(token_count, 1.0),
                f"{prefix}/digit_accuracy": digit_correct / max(digit_total, 1.0),
                f"{prefix}/symbol_accuracy": symbol_correct / max(symbol_total, 1.0),
                f"{prefix}/high_risk_accuracy": risk_correct / max(risk_total, 1.0),
                f"{prefix}/mean_patch_length": mean_patch,
                f"{prefix}/patch_length_std": math.sqrt(patch_variance),
                f"{prefix}/patch_length_min": 0.0 if math.isinf(patch_length_min) else patch_length_min,
                f"{prefix}/patch_length_max": patch_length_max,
                f"{prefix}/mean_boundary_signal": boundary_sum / max(boundary_count, 1.0),
                f"{prefix}/mean_recurrence_update_norm": recurrence_norm_sum / max(
                    recurrence_norm_count, 1.0
                ),
            }
        )
    return result


def write_jsonl(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def train(args: argparse.Namespace, *, patch_mode: str | None = None, output_dir: Path | None = None) -> dict[str, float]:
    seed_everything(args.seed)
    device = torch.device(args.device)
    tokens = load_corpus(args.data)
    if args.validation_data:
        train_tokens = tokens
        validation_tokens = load_corpus(args.validation_data)
    else:
        split = max(int(tokens.numel() * 0.9), args.sequence_length + 2)
        split = min(split, tokens.numel() - args.sequence_length - 2)
        train_tokens, validation_tokens = tokens[:split], tokens[split:]
        if validation_tokens.numel() < args.sequence_length + 1:
            validation_tokens = tokens[-(args.sequence_length + 1) :]

    config = build_config(args, patch_mode=patch_mode)
    model = CARSRModel(config).to(device)
    router_parameters = list(model.patcher.router.parameters())
    router_ids = {id(parameter) for parameter in router_parameters}
    base_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in router_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": base_parameters, "lr": args.learning_rate},
            {
                "params": router_parameters,
                "lr": args.learning_rate * args.router_learning_rate_scale,
                "weight_decay": 0.0,
            },
        ],
        weight_decay=args.weight_decay,
    )
    start_step = 0
    resume = getattr(args, "resume", None)
    if resume:
        loaded, metadata = load_checkpoint(resume, device=device, load_optimizer=True)
        if loaded.config.to_dict() != config.to_dict():
            raise ValueError("resume checkpoint configuration differs from requested run")
        model = loaded
        router_parameters = list(model.patcher.router.parameters())
        router_ids = {id(parameter) for parameter in router_parameters}
        base_parameters = [
            parameter for parameter in model.parameters() if id(parameter) not in router_ids
        ]
        optimizer = torch.optim.AdamW(
            [
                {"params": base_parameters, "lr": args.learning_rate},
                {
                    "params": router_parameters,
                    "lr": args.learning_rate * args.router_learning_rate_scale,
                    "weight_decay": 0.0,
                },
            ],
            weight_decay=args.weight_decay,
        )
        optimizer.load_state_dict(metadata["optimizer"])
        start_step = metadata["step"]

    train_loader = make_loader(
        train_tokens,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        steps=args.steps,
        seed=args.seed,
        shuffle=True,
    )
    validation_loader = make_loader(
        validation_tokens,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        steps=max(args.eval_batches, 1),
        seed=args.seed + 1,
        shuffle=False,
    )
    run_dir = output_dir or Path(args.output)
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    if not resume:
        metrics_path.unlink(missing_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )

    model.train()
    started = time.perf_counter()
    tokens_seen = 0
    last_metrics: dict[str, float] = {}
    last_train_metrics: dict[str, float] = {}
    for offset, batch in enumerate(train_loader, start=1):
        step = start_step + offset
        if offset > args.steps:
            break
        batch = batch.to(device)
        inputs, labels = batch[:, :-1], batch[:, 1:]
        depth = random.choice(config.recurrent_depths)
        optimizer.zero_grad(set_to_none=True)
        output = model(inputs, labels=labels, recurrent_depth=depth)
        assert output.loss is not None and output.token_loss is not None
        warmup = max(args.compression_warmup_steps, 1)
        compression_scale = min(step / warmup, 1.0)
        effective_compression_weight = config.compression_loss_weight * compression_scale
        training_loss = output.token_loss + effective_compression_weight * output.compression_loss
        training_loss.backward()
        router_gradient_norm = math.sqrt(
            sum(
                float(parameter.grad.detach().float().square().sum())
                for parameter in model.patcher.router.parameters()
                if parameter.grad is not None
            )
        )
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()
        tokens_seen += inputs.numel()

        if step % args.log_every == 0 or offset == 1:
            elapsed = max(time.perf_counter() - started, 1e-9)
            lengths = output.patching.patch_lengths[output.patching.patch_mask].float()
            record: dict[str, object] = {
                "step": step,
                "train/loss": float(training_loss.detach()),
                "train/token_loss": float(output.token_loss.detach()),
                "train/compression_loss": float(output.compression_loss.detach()),
                "train/effective_compression_weight": effective_compression_weight,
                "train/bits_per_byte": float(output.token_loss.detach()) / math.log(2.0),
                "train/mean_patch_length": lengths.mean().item(),
                "train/patch_length_std": lengths.std(unbiased=False).item(),
                "train/mean_boundary_signal": output.patching.boundary_probs[
                    inputs.ne(config.pad_token_id)
                ].mean().item(),
                "train/recurrent_depth": depth,
                "train/mean_recurrence_update_norm": (
                    output.recurrence_update_norms.mean().item()
                    if output.recurrence_update_norms.numel()
                    else 0.0
                ),
                "train/gradient_norm": float(gradient_norm),
                "train/router_gradient_norm": router_gradient_norm,
                "train/tokens_per_second": tokens_seen / elapsed,
            }
            last_train_metrics = {
                key: float(value)
                for key, value in record.items()
                if key != "step" and isinstance(value, (int, float))
            }
            write_jsonl(metrics_path, record)
            print(json.dumps(record, sort_keys=True))

        if step % args.eval_every == 0 or offset == args.steps:
            last_metrics = evaluate(
                model,
                validation_loader,
                device=device,
                depths=config.recurrent_depths,
                max_batches=args.eval_batches,
            )
            record = {"step": step, **last_metrics}
            write_jsonl(metrics_path, record)
            print(json.dumps(record, sort_keys=True))
            model.train()
            save_checkpoint(
                run_dir / "checkpoint.pt",
                model,
                optimizer=optimizer,
                step=step,
                metrics=last_metrics,
            )
    summary = {
        "parameters": float(model.parameter_count),
        "steps": float(start_step + args.steps),
        "allocated_steps": float(args.steps),
        "training_bytes": float(train_tokens.numel()),
        "validation_bytes": float(validation_tokens.numel()),
        **last_train_metrics,
        **last_metrics,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def run_evaluate(args: argparse.Namespace) -> None:
    model, metadata = load_checkpoint(args.checkpoint, device=args.device)
    tokens = load_corpus(args.data)
    loader = make_loader(
        tokens,
        sequence_length=model.config.max_seq_len,
        batch_size=args.batch_size,
        steps=args.eval_batches,
        seed=args.seed,
        shuffle=False,
    )
    metrics = evaluate(
        model,
        loader,
        device=torch.device(args.device),
        depths=model.config.recurrent_depths,
        max_batches=args.eval_batches,
    )
    print(json.dumps({"checkpoint_step": metadata["step"], **metrics}, indent=2, sort_keys=True))


def run_generate(args: argparse.Namespace) -> None:
    model, _ = load_checkpoint(args.checkpoint, device=args.device)
    model.eval()
    tokenizer = ByteTokenizer()
    prompt = torch.tensor(
        [tokenizer.encode(args.prompt, add_eos=False)],
        device=args.device,
        dtype=torch.long,
    )
    generated = model.generate(
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        recurrent_depth=args.depth,
    )
    print(tokenizer.decode(generated[0].tolist()))


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def run_benchmark(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    model, metadata = load_checkpoint(args.checkpoint, device=device)
    model.eval()
    tokenizer = ByteTokenizer()
    prompt_tokens = tokenizer.encode(args.prompt, add_eos=False)
    room = model.config.max_seq_len - args.decode_steps
    if room < 1:
        raise ValueError("decode_steps leaves no room for a prompt")
    prompt_tokens = prompt_tokens[-room:]
    prompt = torch.tensor([prompt_tokens], device=device, dtype=torch.long)
    depth = model.config.default_recurrent_depth if args.depth is None else args.depth

    for _ in range(args.warmup):
        model(prompt, recurrent_depth=depth)
        model.start_generation_session(prompt, recurrent_depth=depth)
    _synchronize(device)

    prefill_samples: list[float] = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        session = model.start_generation_session(prompt, recurrent_depth=depth)
        _synchronize(device)
        prefill_samples.append((time.perf_counter() - started) * 1000.0)

    decode_samples: list[float] = []
    cache_bytes = 0
    for _ in range(args.repeats):
        session = model.start_generation_session(prompt, recurrent_depth=depth)
        started = time.perf_counter()
        for _ in range(args.decode_steps):
            token = session.sample(temperature=0.0)
            session.append(token)
        _synchronize(device)
        elapsed = time.perf_counter() - started
        decode_samples.append(elapsed * 1000.0 / max(args.decode_steps, 1))
        cache_bytes = session.state.cache_bytes

    output = model(prompt, recurrent_depth=depth)
    report: dict[str, object] = {
        "checkpoint_step": metadata["step"],
        "device": str(device),
        "model_dtype": str(model.token_embedding.weight.dtype),
        "cache_dtype": model.config.cache_dtype,
        "patch_attention": model.config.patch_attention,
        "cpla_content_dim": model.config.cpla_content_dim,
        "cpla_position_dim": model.config.cpla_position_dim,
        "parameters": model.parameter_count,
        "recurrent_depth": depth,
        "prompt_bytes": prompt.shape[1],
        "mean_patch_length": output.patching.patch_lengths[
            output.patching.patch_mask
        ].float().mean().item(),
        "patch_count": output.patching.patch_count.item(),
        "prefill_ms_median": statistics.median(prefill_samples),
        "prefill_ms_min": min(prefill_samples),
        "decode_ms_per_byte_median": statistics.median(decode_samples),
        "decode_ms_per_byte_min": min(decode_samples),
        "hierarchical_cache_bytes": cache_bytes,
    }
    if device.type == "cuda":
        report["cuda_peak_memory_bytes"] = torch.cuda.max_memory_allocated(device)
    print(json.dumps(report, indent=2, sort_keys=True))



_ALLOWED_ARM_OVERRIDES = {
    "patch_mode",
    "patch_attention",
    "patch_ratio",
    "min_patch_size",
    "max_patch_size",
    "cpla_content_dim",
    "cpla_position_dim",
    "disable_span_mass_bias",
    "depths",
    "default_depth",
}


def _safe_arm_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip("-.")
    if not safe:
        raise ValueError("Thompson arm name must contain a filesystem-safe character")
    return safe


def _validate_arm_overrides(overrides: dict[str, Any]) -> dict[str, Any]:
    unknown = set(overrides) - _ALLOWED_ARM_OVERRIDES
    if unknown:
        raise ValueError(f"unsupported Thompson arm overrides: {sorted(unknown)}")
    normalized = dict(overrides)
    if "depths" in normalized:
        normalized["depths"] = [int(value) for value in normalized["depths"]]
    for key in (
        "patch_ratio",
        "min_patch_size",
        "max_patch_size",
        "cpla_content_dim",
        "cpla_position_dim",
        "default_depth",
    ):
        if key in normalized:
            normalized[key] = int(normalized[key])
    if "disable_span_mass_bias" in normalized:
        normalized["disable_span_mass_bias"] = bool(normalized["disable_span_mass_bias"])
    return normalized


def built_in_thompson_arms(args: argparse.Namespace) -> list[tuple[str, dict[str, Any]]]:
    if args.study == "core":
        return [
            (
                "byte-gqa",
                {
                    "patch_mode": "none",
                    "patch_attention": "gqa",
                    "depths": [0],
                    "default_depth": 0,
                },
            ),
            (
                "fixed-gqa",
                {
                    "patch_mode": "fixed",
                    "patch_attention": "gqa",
                    "depths": [0],
                    "default_depth": 0,
                },
            ),
            (
                "learned-gqa",
                {
                    "patch_mode": "learned",
                    "patch_attention": "gqa",
                    "depths": [0],
                    "default_depth": 0,
                },
            ),
            (
                "learned-cpla",
                {
                    "patch_mode": "learned",
                    "patch_attention": "cpla",
                    "depths": [0],
                    "default_depth": 0,
                },
            ),
            (
                "learned-cpla-recurrent",
                {
                    "patch_mode": "learned",
                    "patch_attention": "cpla",
                    "depths": list(args.depths),
                    "default_depth": args.default_depth,
                },
            ),
        ]
    if args.study == "patch":
        return [
            (
                mode,
                {
                    "patch_mode": mode,
                    "patch_attention": args.patch_attention,
                    "depths": list(args.depths),
                    "default_depth": args.default_depth,
                },
            )
            for mode in ("none", "fixed", "learned")
        ]
    if args.study == "attention":
        return [
            (
                attention,
                {
                    "patch_mode": "learned",
                    "patch_attention": attention,
                    "depths": list(args.depths),
                    "default_depth": args.default_depth,
                },
            )
            for attention in ("gqa", "cpla")
        ]
    if args.study == "cpla-rank":
        return [
            (
                f"cpla-{rank}",
                {
                    "patch_mode": "learned",
                    "patch_attention": "cpla",
                    "cpla_content_dim": rank,
                    "depths": list(args.depths),
                    "default_depth": args.default_depth,
                },
            )
            for rank in (32, 48, 64)
        ]
    if args.study == "depth":
        return [
            (
                f"depth-{depth}",
                {
                    "patch_mode": "learned",
                    "patch_attention": args.patch_attention,
                    "depths": [depth],
                    "default_depth": depth,
                },
            )
            for depth in (0, 1, 2, 4)
        ]
    raise ValueError(f"unknown Thompson study: {args.study}")


def load_thompson_arm_specs(args: argparse.Namespace) -> list[tuple[str, dict[str, Any]]]:
    if args.arms_config:
        values = json.loads(Path(args.arms_config).read_text(encoding="utf-8"))
        if not isinstance(values, list) or not values:
            raise ValueError("arms config must be a non-empty JSON list")
        specs: list[tuple[str, dict[str, Any]]] = []
        for item in values:
            if not isinstance(item, dict) or "name" not in item:
                raise ValueError("each arm must contain name and optional overrides")
            specs.append(
                (
                    _safe_arm_name(str(item["name"])),
                    _validate_arm_overrides(dict(item.get("overrides", {}))),
                )
            )
    else:
        specs = [
            (_safe_arm_name(name), _validate_arm_overrides(overrides))
            for name, overrides in built_in_thompson_arms(args)
        ]
    names = [name for name, _ in specs]
    if len(names) != len(set(names)):
        raise ValueError("Thompson arm names must be unique")
    return specs


def apply_arm_overrides(
    args: argparse.Namespace,
    overrides: dict[str, Any],
) -> argparse.Namespace:
    arm_args = deepcopy(args)
    for key, value in overrides.items():
        setattr(arm_args, key, deepcopy(value))
    return arm_args


def patch_cache_ratio(config: CARSRConfig, mean_patch_length: float) -> float:
    head_dim = config.d_model // config.n_heads
    gqa_values = 2 * config.n_kv_heads * head_dim
    if config.patch_attention == "cpla":
        row_values = config.cpla_content_dim + config.cpla_position_dim
    else:
        row_values = gqa_values
    return (row_values / gqa_values) / max(mean_patch_length, 1.0)


def compute_research_reward(
    summary: dict[str, float],
    config: CARSRConfig,
    *,
    exact_weight: float,
    cache_weight: float,
    depth_weight: float,
    throughput_weight: float,
) -> tuple[float, dict[str, float]]:
    prefix = f"depth_{config.default_recurrent_depth}"
    bpb_key = f"{prefix}/bits_per_byte"
    exact_key = f"{prefix}/high_risk_accuracy"
    patch_key = f"{prefix}/mean_patch_length"
    if bpb_key not in summary:
        raise ValueError(f"training summary does not contain {bpb_key}")
    bits_per_byte = float(summary[bpb_key])
    exact_accuracy = float(summary.get(exact_key, 0.0))
    mean_patch_length = float(summary.get(patch_key, 1.0))
    cache_ratio = patch_cache_ratio(config, mean_patch_length)
    depth_cost = float(config.default_recurrent_depth)
    throughput = max(float(summary.get("train/tokens_per_second", 0.0)), 0.0)
    throughput_score = math.log1p(throughput)
    components = {
        "quality": -bits_per_byte,
        "exact_penalty": -exact_weight * (1.0 - exact_accuracy),
        "cache_penalty": -cache_weight * cache_ratio,
        "depth_penalty": -depth_weight * depth_cost,
        "throughput_bonus": throughput_weight * throughput_score,
        "bits_per_byte": bits_per_byte,
        "high_risk_accuracy": exact_accuracy,
        "cache_ratio": cache_ratio,
        "default_recurrent_depth": depth_cost,
        "tokens_per_second": throughput,
    }
    reward = sum(
        components[key]
        for key in (
            "quality",
            "exact_penalty",
            "cache_penalty",
            "depth_penalty",
            "throughput_bonus",
        )
    )
    components["reward"] = reward
    return reward, components


def thompson_study_metadata(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "study": args.study,
        "seed": args.seed,
        "min_pulls": args.min_pulls,
        "arms_config": (
            None if not args.arms_config else str(Path(args.arms_config).resolve())
        ),
        "base_model_config": build_config(args).to_dict(),
        "training": {
            "data": None if args.data is None else str(Path(args.data).resolve()),
            "validation_data": (
                None
                if args.validation_data is None
                else str(Path(args.validation_data).resolve())
            ),
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "router_learning_rate_scale": args.router_learning_rate_scale,
            "compression_warmup_steps": args.compression_warmup_steps,
            "grad_clip": args.grad_clip,
            "eval_batches": args.eval_batches,
            "round_steps": args.round_steps,
        },
        "reward": {
            "exact_weight": args.reward_exact_weight,
            "cache_weight": args.reward_cache_weight,
            "depth_weight": args.reward_depth_weight,
            "throughput_weight": args.reward_throughput_weight,
        },
        "prior": {
            "mean": args.prior_mean,
            "strength": args.prior_strength,
            "alpha": args.prior_alpha,
            "beta": args.prior_beta,
        },
    }


def _new_thompson_scheduler(
    args: argparse.Namespace,
    specs: list[tuple[str, dict[str, Any]]],
) -> ThompsonResearchScheduler:
    arms = [
        ThompsonArm(
            name=name,
            overrides=overrides,
            posterior=NormalInverseGammaPosterior(
                prior_mean=args.prior_mean,
                prior_strength=args.prior_strength,
                prior_alpha=args.prior_alpha,
                prior_beta=args.prior_beta,
            ),
        )
        for name, overrides in specs
    ]
    return ThompsonResearchScheduler(
        arms,
        seed=args.seed,
        min_pulls=args.min_pulls,
        metadata=thompson_study_metadata(args),
    )


def _verify_scheduler_specs(
    scheduler: ThompsonResearchScheduler,
    specs: list[tuple[str, dict[str, Any]]],
    args: argparse.Namespace,
) -> None:
    expected = {name: overrides for name, overrides in specs}
    actual = {name: arm.overrides for name, arm in scheduler.arms.items()}
    if expected != actual:
        raise ValueError(
            "existing Thompson state uses different arms; use --reset-scheduler or a new output directory"
        )
    if scheduler.metadata != thompson_study_metadata(args):
        raise ValueError(
            "existing Thompson state uses different shared settings or reward coefficients; "
            "use --reset-scheduler or a new output directory"
        )


def run_thompson(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "thompson_state.json"
    allocation_path = root / "allocations.jsonl"
    specs = load_thompson_arm_specs(args)

    if state_path.exists() and not args.reset_scheduler:
        scheduler = ThompsonResearchScheduler.load(state_path)
        _verify_scheduler_specs(scheduler, specs, args)
    else:
        if args.reset_scheduler:
            state_path.unlink(missing_ok=True)
            allocation_path.unlink(missing_ok=True)
        scheduler = _new_thompson_scheduler(args, specs)
        scheduler.save(state_path)

    for _ in range(args.rounds):
        arm, posterior_samples = scheduler.select()
        arm_dir = root / "arms" / arm.name
        arm_args = apply_arm_overrides(args, arm.overrides)
        arm_args.steps = args.round_steps
        arm_args.output = str(arm_dir)
        arm_args.seed = args.seed + arm.pulls * 10_007
        checkpoint = arm_dir / "checkpoint.pt"
        arm_args.resume = str(checkpoint) if checkpoint.exists() else None
        arm_args.eval_every = min(args.eval_every, args.round_steps)
        print(
            json.dumps(
                {
                    "thompson/round": scheduler.round_index,
                    "thompson/selected_arm": arm.name,
                    "thompson/posterior_samples": posterior_samples,
                    "thompson/overrides": arm.overrides,
                    "thompson/resume": arm_args.resume,
                },
                sort_keys=True,
            )
        )
        summary = train(arm_args, output_dir=arm_dir)
        config = build_config(arm_args)
        reward, components = compute_research_reward(
            summary,
            config,
            exact_weight=args.reward_exact_weight,
            cache_weight=args.reward_cache_weight,
            depth_weight=args.reward_depth_weight,
            throughput_weight=args.reward_throughput_weight,
        )
        round_index = scheduler.round_index
        scheduler.update(
            arm.name,
            reward=reward,
            allocated_steps=args.round_steps,
            summary=summary,
        )
        record: dict[str, Any] = {
            "round": round_index,
            "selected_arm": arm.name,
            "posterior_samples": posterior_samples,
            "overrides": arm.overrides,
            "reward_components": components,
            "posterior_after": scheduler.arms[arm.name].posterior.to_dict(),
            "pulls_after": scheduler.arms[arm.name].pulls,
            "allocated_steps_after": scheduler.arms[arm.name].allocated_steps,
            "arm_seed": arm_args.seed,
            "checkpoint": str(checkpoint),
        }
        write_jsonl(allocation_path, record)
        scheduler.save(state_path)
        print(json.dumps({"thompson/update": record}, sort_keys=True))

    report: dict[str, Any] = {
        "version": THOMPSON_STATE_VERSION,
        "rounds_completed": scheduler.round_index,
        "reward": {
            "exact_weight": args.reward_exact_weight,
            "cache_weight": args.reward_cache_weight,
            "depth_weight": args.reward_depth_weight,
            "throughput_weight": args.reward_throughput_weight,
        },
        "arms": {
            name: {
                "overrides": arm.overrides,
                "pulls": arm.pulls,
                "allocated_steps": arm.allocated_steps,
                "last_reward": arm.last_reward,
                "posterior_mean": arm.posterior.posterior_mean,
                "posterior_std": arm.posterior.posterior_std,
                "last_summary": arm.last_summary,
            }
            for name, arm in sorted(scheduler.arms.items())
        },
    }
    (root / "thompson_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def run_ablation(args: argparse.Namespace) -> None:
    root = Path(args.output)
    summaries: dict[str, dict[str, float]] = {}
    for mode in ("none", "fixed", "learned"):
        print(f"\n=== ablation: {mode} ===")
        summaries[mode] = train(
            args,
            patch_mode=mode,
            output_dir=root / mode,
        )
    (root / "ablation_summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summaries, indent=2, sort_keys=True))


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", help="UTF-8 training corpus; omitted uses a tiny smoke corpus")
    parser.add_argument("--validation-data", help="separate UTF-8 validation corpus; otherwise a 90/10 development split is used")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=192)
    parser.add_argument("--n-heads", type=int, default=6)
    parser.add_argument("--n-kv-heads", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=576)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--byte-layers", type=int, default=2)
    parser.add_argument("--byte-window", type=int, default=128)
    parser.add_argument("--patch-layers", type=int, default=4)
    parser.add_argument("--decoder-layers", type=int, default=1)
    parser.add_argument("--patch-mode", choices=("learned", "fixed", "none"), default="learned")
    parser.add_argument("--patch-attention", choices=("cpla", "gqa"), default="cpla")
    parser.add_argument("--patch-ratio", type=int, default=4)
    parser.add_argument("--min-patch-size", type=int, default=2)
    parser.add_argument("--max-patch-size", type=int, default=8)
    parser.add_argument("--compression-weight", type=float, default=0.03)
    parser.add_argument("--cpla-content-dim", type=int, default=48)
    parser.add_argument("--cpla-position-dim", type=int, default=16)
    parser.add_argument("--disable-span-mass-bias", action="store_true")
    parser.add_argument("--depths", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--default-depth", type=int, default=2)
    parser.add_argument("--cache-dtype", choices=("model", "float16", "bfloat16"), default="model")


def add_training_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_output: str,
    include_resume: bool = False,
) -> None:
    parser.add_argument("--output", default=default_output)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--router-learning-rate-scale", type=float, default=1.0)
    parser.add_argument("--compression-warmup-steps", type=int, default=1000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=16)
    if include_resume:
        parser.add_argument("--resume")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CARS-R research experiments")
    sub = parser.add_subparsers(dest="command", required=True)

    train_parser = sub.add_parser("train", help="train one research variant")
    add_model_arguments(train_parser)
    add_training_arguments(
        train_parser, default_output="runs/learned", include_resume=True
    )
    train_parser.add_argument("--steps", type=int, default=1000)
    train_parser.set_defaults(function=train)

    ablate_parser = sub.add_parser("ablate", help="train none/fixed/learned patch controls")
    add_model_arguments(ablate_parser)
    add_training_arguments(ablate_parser, default_output="runs/ablation")
    ablate_parser.add_argument("--steps", type=int, default=1000)
    ablate_parser.set_defaults(function=run_ablation)

    thompson_parser = sub.add_parser(
        "thompson",
        help="allocate incremental training rounds with resumable Thompson sampling",
    )
    add_model_arguments(thompson_parser)
    add_training_arguments(thompson_parser, default_output="runs/thompson")
    thompson_parser.add_argument(
        "--study",
        choices=("core", "patch", "attention", "cpla-rank", "depth"),
        default="core",
        help="built-in arm set; ignored when --arms-config is provided",
    )
    thompson_parser.add_argument(
        "--arms-config",
        help="optional JSON list of named architecture overrides",
    )
    thompson_parser.add_argument("--rounds", type=int, default=10)
    thompson_parser.add_argument("--round-steps", type=int, default=1000)
    thompson_parser.add_argument("--min-pulls", type=int, default=1)
    thompson_parser.add_argument("--reset-scheduler", action="store_true")
    thompson_parser.add_argument("--prior-mean", type=float, default=-8.0)
    thompson_parser.add_argument("--prior-strength", type=float, default=0.25)
    thompson_parser.add_argument("--prior-alpha", type=float, default=2.0)
    thompson_parser.add_argument("--prior-beta", type=float, default=1.0)
    thompson_parser.add_argument("--reward-exact-weight", type=float, default=0.05)
    thompson_parser.add_argument("--reward-cache-weight", type=float, default=0.02)
    thompson_parser.add_argument("--reward-depth-weight", type=float, default=0.005)
    thompson_parser.add_argument("--reward-throughput-weight", type=float, default=0.0)
    thompson_parser.set_defaults(function=run_thompson)

    evaluate_parser = sub.add_parser("evaluate", help="evaluate all trained depths")
    evaluate_parser.add_argument("checkpoint")
    evaluate_parser.add_argument("--data")
    evaluate_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    evaluate_parser.add_argument("--batch-size", type=int, default=8)
    evaluate_parser.add_argument("--eval-batches", type=int, default=32)
    evaluate_parser.add_argument("--seed", type=int, default=23)
    evaluate_parser.set_defaults(function=run_evaluate)

    generate_parser = sub.add_parser("generate", help="generate with hierarchical KV caches")
    generate_parser.add_argument("checkpoint")
    generate_parser.add_argument("prompt")
    generate_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    generate_parser.add_argument("--max-new-tokens", type=int, default=64)
    generate_parser.add_argument("--temperature", type=float, default=0.0)
    generate_parser.add_argument("--top-k", type=int)
    generate_parser.add_argument("--depth", type=int)
    generate_parser.set_defaults(function=run_generate)

    benchmark_parser = sub.add_parser(
        "benchmark", help="measure prefill, decode, and hierarchical cache memory"
    )
    benchmark_parser.add_argument("checkpoint")
    benchmark_parser.add_argument("--prompt", default="The central result is")
    benchmark_parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    benchmark_parser.add_argument("--depth", type=int)
    benchmark_parser.add_argument("--decode-steps", type=int, default=32)
    benchmark_parser.add_argument("--warmup", type=int, default=2)
    benchmark_parser.add_argument("--repeats", type=int, default=5)
    benchmark_parser.set_defaults(function=run_benchmark)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = args.function(args)
    if isinstance(result, dict):
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

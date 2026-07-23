from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Literal, Protocol

import torch
from torch import Tensor, nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Configuration and byte tokenization
# -----------------------------------------------------------------------------


@dataclass(slots=True)
class CARSRConfig:
    """CARS-R 0.1.0 research configuration.

    The model combines bounded local byte processing, causal byte-to-patch
    compression, patch-aware global attention, and shared recurrent refinement.
    CPLA is the primary patch-attention backend; matched GQA remains available
    as the controlled baseline required for scientific comparison.
    """

    vocab_size: int = 259
    d_model: int = 192
    n_heads: int = 6
    n_kv_heads: int = 2
    d_ff: int = 576
    dropout: float = 0.0

    byte_layers: int = 2
    byte_window: int = 128
    patch_layers: int = 4
    decoder_layers: int = 1
    max_seq_len: int = 2048

    patch_mode: Literal["learned", "fixed", "none"] = "learned"
    patch_attention: Literal["cpla", "gqa"] = "cpla"
    target_patch_ratio: int = 4
    min_patch_size: int = 2
    max_patch_size: int = 8
    assignment_temperature: float = 0.25
    compression_loss_weight: float = 0.03

    cpla_content_dim: int = 48
    cpla_position_dim: int = 16
    span_mass_bias: bool = True

    recurrent_depths: tuple[int, ...] = (1, 2, 4)
    default_recurrent_depth: int = 2

    tie_embeddings: bool = True
    cache_dtype: Literal["model", "float16", "bfloat16"] = "model"

    pad_token_id: int = 256
    bos_token_id: int = 257
    eos_token_id: int = 258

    def validate(self) -> None:
        if self.vocab_size < 259:
            raise ValueError("vocab_size must include 256 bytes plus PAD/BOS/EOS")
        if self.d_model <= 0 or self.d_ff <= 0:
            raise ValueError("model dimensions must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.byte_layers < 1 or self.patch_layers < 1 or self.decoder_layers < 0:
            raise ValueError("invalid layer counts")
        if self.byte_window < 1 or self.max_seq_len < 2:
            raise ValueError("invalid sequence limits")
        if self.patch_mode not in {"learned", "fixed", "none"}:
            raise ValueError("patch_mode must be learned, fixed, or none")
        if self.patch_attention not in {"cpla", "gqa"}:
            raise ValueError("patch_attention must be cpla or gqa")
        if self.target_patch_ratio < 1:
            raise ValueError("target_patch_ratio must be positive")
        if self.min_patch_size < 1 or self.max_patch_size < self.min_patch_size:
            raise ValueError("invalid patch-size bounds")
        if self.patch_mode == "fixed" and not (
            self.min_patch_size <= self.target_patch_ratio <= self.max_patch_size
        ):
            raise ValueError("fixed patch ratio must lie inside patch-size bounds")
        if self.assignment_temperature <= 0:
            raise ValueError("assignment_temperature must be positive")
        if self.compression_loss_weight < 0:
            raise ValueError("compression_loss_weight cannot be negative")
        if self.cpla_content_dim < 4:
            raise ValueError("cpla_content_dim must be at least four")
        if self.cpla_position_dim < 8 or self.cpla_position_dim % 4:
            raise ValueError("cpla_position_dim must be divisible by four and at least eight")
        if not self.recurrent_depths or min(self.recurrent_depths) < 0:
            raise ValueError("recurrent_depths must contain non-negative depths")
        if self.default_recurrent_depth not in self.recurrent_depths:
            raise ValueError("default_recurrent_depth must be in recurrent_depths")
        if self.cache_dtype not in {"model", "float16", "bfloat16"}:
            raise ValueError("invalid cache_dtype")
        special = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
        if len(special) != 3 or min(special) < 0 or max(special) >= self.vocab_size:
            raise ValueError("invalid special token IDs")

    @property
    def max_patches(self) -> int:
        if self.patch_mode == "none":
            return self.max_seq_len
        return math.ceil(self.max_seq_len / self.min_patch_size) + 1

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["recurrent_depths"] = list(self.recurrent_depths)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CARSRConfig":
        clean = dict(data)
        if "recurrent_depths" in clean:
            clean["recurrent_depths"] = tuple(clean["recurrent_depths"])
        config = cls(**clean)
        config.validate()
        return config


class ByteTokenizer:
    pad_token_id = 256
    bos_token_id = 257
    eos_token_id = 258
    vocab_size = 259

    def encode(self, text: str, *, add_eos: bool = True) -> list[int]:
        tokens = [self.bos_token_id, *text.encode("utf-8")]
        if add_eos:
            tokens.append(self.eos_token_id)
        return tokens

    def decode(self, tokens: list[int]) -> str:
        return bytes(token for token in tokens if 0 <= token < 256).decode(
            "utf-8", errors="replace"
        )


# -----------------------------------------------------------------------------
# Core neural layers
# -----------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        source_dtype = x.dtype
        x32 = x.float()
        normalized = x32 * torch.rsqrt(x32.square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(source_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.out_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        gate, value = self.in_proj(x).chunk(2, dim=-1)
        return self.out_proj(self.dropout(F.silu(gate) * value))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: float = 10_000.0) -> None:
        super().__init__()
        if dim % 2:
            raise ValueError("RoPE dimension must be even")
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        angles = positions.float().unsqueeze(-1) * self.inv_freq
        return angles.cos().to(dtype), angles.sin().to(dtype)


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # x: [batch, heads, time, dim], cos/sin: [batch|1, time, dim/2]
    even, odd = x[..., 0::2], x[..., 1::2]
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class DualSpanRoPE(nn.Module):
    """RoPE over patch order and original byte geometry.

    Half of the positional channels encode patch ordinal position. The other
    half encodes the patch centre in original byte coordinates. This avoids the
    uniform-token assumption that every patch occupies one equal-width step.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim % 4:
            raise ValueError("dual span RoPE dimension must be divisible by four")
        self.half = dim // 2
        self.patch_rope = RotaryEmbedding(self.half)
        self.byte_rope = RotaryEmbedding(self.half)

    def forward(
        self,
        x: Tensor,
        patch_positions: Tensor,
        byte_centres: Tensor,
    ) -> Tensor:
        first, second = x.split(self.half, dim=-1)
        p_cos, p_sin = self.patch_rope(patch_positions, x.dtype)
        b_cos, b_sin = self.byte_rope(byte_centres, x.dtype)
        return torch.cat(
            (_apply_rope(first, p_cos, p_sin), _apply_rope(second, b_cos, b_sin)),
            dim=-1,
        )


# -----------------------------------------------------------------------------
# Cache structures
# -----------------------------------------------------------------------------


@dataclass
class KVCache:
    key: Tensor
    value: Tensor
    length: int = 0
    next_index: int = 0

    @classmethod
    def allocate(
        cls,
        *,
        batch: int,
        n_kv_heads: int,
        capacity: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "KVCache":
        shape = (batch, n_kv_heads, capacity, head_dim)
        return cls(
            key=torch.empty(shape, device=device, dtype=dtype),
            value=torch.empty(shape, device=device, dtype=dtype),
        )

    @property
    def capacity(self) -> int:
        return self.key.shape[2]

    def append(self, key: Tensor, value: Tensor) -> None:
        if key.shape[2] != 1 or value.shape != key.shape:
            raise ValueError("cache append expects one aligned K/V position")
        self.key[:, :, self.next_index : self.next_index + 1].copy_(key.to(self.key.dtype))
        self.value[:, :, self.next_index : self.next_index + 1].copy_(value.to(self.value.dtype))
        self.next_index = (self.next_index + 1) % self.capacity
        self.length = min(self.length + 1, self.capacity)

    def _indices(self) -> Tensor:
        if self.length == 0:
            raise RuntimeError("cannot read an empty cache")
        if self.length < self.capacity:
            return torch.arange(self.length, device=self.key.device)
        return torch.cat(
            (
                torch.arange(self.next_index, self.capacity, device=self.key.device),
                torch.arange(0, self.next_index, device=self.key.device),
            )
        )

    def ordered(self, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        indices = self._indices()
        return self.key.index_select(2, indices).to(dtype), self.value.index_select(2, indices).to(dtype)

    @property
    def bytes(self) -> int:
        return self.key.numel() * self.key.element_size() + self.value.numel() * self.value.element_size()


@dataclass
class CPLACache:
    latent: Tensor
    position_key: Tensor
    lengths: Tensor
    length: int = 0
    next_index: int = 0

    @classmethod
    def allocate(
        cls,
        *,
        batch: int,
        capacity: int,
        content_dim: int,
        position_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "CPLACache":
        return cls(
            latent=torch.empty(batch, capacity, content_dim, device=device, dtype=dtype),
            position_key=torch.empty(batch, capacity, position_dim, device=device, dtype=dtype),
            lengths=torch.empty(batch, capacity, device=device, dtype=torch.long),
        )

    @property
    def capacity(self) -> int:
        return self.latent.shape[1]

    def append(self, latent: Tensor, position_key: Tensor, lengths: Tensor) -> None:
        if latent.ndim != 3 or latent.shape[1] != 1:
            raise ValueError("CPLA cache append expects one position")
        self.latent[:, self.next_index : self.next_index + 1].copy_(latent.to(self.latent.dtype))
        self.position_key[:, self.next_index : self.next_index + 1].copy_(
            position_key.to(self.position_key.dtype)
        )
        self.lengths[:, self.next_index : self.next_index + 1].copy_(lengths[:, None])
        self.next_index = (self.next_index + 1) % self.capacity
        self.length = min(self.length + 1, self.capacity)

    def _indices(self) -> Tensor:
        if self.length == 0:
            raise RuntimeError("cannot read an empty CPLA cache")
        if self.length < self.capacity:
            return torch.arange(self.length, device=self.latent.device)
        return torch.cat(
            (
                torch.arange(self.next_index, self.capacity, device=self.latent.device),
                torch.arange(0, self.next_index, device=self.latent.device),
            )
        )

    def ordered(self, dtype: torch.dtype) -> tuple[Tensor, Tensor, Tensor]:
        indices = self._indices()
        return (
            self.latent.index_select(1, indices).to(dtype),
            self.position_key.index_select(1, indices).to(dtype),
            self.lengths.index_select(1, indices),
        )

    @property
    def bytes(self) -> int:
        return sum(
            tensor.numel() * tensor.element_size()
            for tensor in (self.latent, self.position_key, self.lengths)
        )


AttentionCache = KVCache | CPLACache


# -----------------------------------------------------------------------------
# Local byte GQA
# -----------------------------------------------------------------------------


class GQAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryEmbedding(self.head_dim)
        self.dropout = dropout

    def _project(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch, time, _ = x.shape
        query = self.q_proj(x).view(batch, time, self.n_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(x).view(batch, time, self.n_kv_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(x).view(batch, time, self.n_kv_heads, self.head_dim).transpose(1, 2)
        return query, key, value

    def _query(self, x: Tensor) -> Tensor:
        batch, time, _ = x.shape
        return self.q_proj(x).view(batch, time, self.n_heads, self.head_dim).transpose(1, 2)

    def _key_value(self, x: Tensor) -> tuple[Tensor, Tensor]:
        batch, time, _ = x.shape
        key = self.k_proj(x).view(batch, time, self.n_kv_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(x).view(batch, time, self.n_kv_heads, self.head_dim).transpose(1, 2)
        return key, value

    def _repeat_kv(self, x: Tensor) -> Tensor:
        repeat = self.n_heads // self.n_kv_heads
        return x if repeat == 1 else x.repeat_interleave(repeat, dim=1)

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        positions: Tensor,
        *,
        local_window: int | None,
    ) -> Tensor:
        batch, time, _ = x.shape
        query, key, value = self._project(x)
        if positions.ndim == 1:
            positions = positions.unsqueeze(0).expand(batch, -1)
        cos, sin = self.rope(positions, query.dtype)
        query = _apply_rope(query, cos, sin)
        key = _apply_rope(key, cos, sin)

        if local_window is not None:
            key = self._repeat_kv(key)
            value = self._repeat_kv(value)
            window = min(local_window, time)
            padded_key = F.pad(key, (0, 0, window - 1, 0))
            padded_value = F.pad(value, (0, 0, window - 1, 0))
            key_windows = padded_key.unfold(2, window, 1).permute(0, 1, 2, 4, 3)
            value_windows = padded_value.unfold(2, window, 1).permute(0, 1, 2, 4, 3)
            key_mask = F.pad(mask, (window - 1, 0), value=False).unfold(1, window, 1)
            safe_mask = torch.where(
                mask.unsqueeze(-1),
                key_mask,
                F.one_hot(
                    torch.full((batch, time), window - 1, device=x.device, dtype=torch.long),
                    window,
                ).bool(),
            )
            scores = torch.einsum("bhtd,bhtwd->bhtw", query, key_windows)
            scores = scores * (self.head_dim**-0.5)
            scores = scores.masked_fill(~safe_mask[:, None], torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
            weights = F.dropout(weights, p=self.dropout, training=self.training)
            output = torch.einsum("bhtw,bhtwd->bhtd", weights, value_windows)
        else:
            output = self._full_attention(query, key, value, mask, mask, positions, positions)
        output = output.transpose(1, 2).contiguous().view(batch, time, self.d_model)
        return self.out_proj(output) * mask.unsqueeze(-1).to(output.dtype)

    def _full_attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        query_mask: Tensor,
        memory_mask: Tensor,
        query_positions: Tensor,
        memory_positions: Tensor,
    ) -> Tensor:
        if query_positions.ndim == 1:
            query_positions = query_positions.unsqueeze(0).expand(query.shape[0], -1)
        if memory_positions.ndim == 1:
            memory_positions = memory_positions.unsqueeze(0).expand(query.shape[0], -1)
        allowed = memory_positions[:, None, :] <= query_positions[:, :, None]
        allowed &= memory_mask[:, None, :]
        safe = torch.where(
            query_mask[:, :, None],
            allowed,
            F.one_hot(
                torch.zeros(query.shape[0], query.shape[2], device=query.device, dtype=torch.long),
                key.shape[2],
            ).bool(),
        )
        scores = torch.einsum("bhtd,bhsd->bhts", query, self._repeat_kv(key))
        scores = scores * (self.head_dim**-0.5)
        scores = scores.masked_fill(~safe[:, None], torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        return torch.einsum("bhts,bhsd->bhtd", weights, self._repeat_kv(value))

    def forward_cross(
        self,
        query_x: Tensor,
        memory_x: Tensor,
        query_mask: Tensor,
        memory_mask: Tensor,
        query_positions: Tensor,
        memory_positions: Tensor,
    ) -> Tensor:
        query = self._query(query_x)
        key, value = self._key_value(memory_x)
        if query_positions.ndim == 1:
            query_positions = query_positions.unsqueeze(0).expand(query_x.shape[0], -1)
        if memory_positions.ndim == 1:
            memory_positions = memory_positions.unsqueeze(0).expand(memory_x.shape[0], -1)
        q_cos, q_sin = self.rope(query_positions, query.dtype)
        k_cos, k_sin = self.rope(memory_positions, key.dtype)
        query = _apply_rope(query, q_cos, q_sin)
        key = _apply_rope(key, k_cos, k_sin)
        output = self._full_attention(
            query,
            key,
            value,
            query_mask,
            memory_mask,
            query_positions,
            memory_positions,
        )
        output = output.transpose(1, 2).contiguous().view_as(query_x)
        return self.out_proj(output) * query_mask.unsqueeze(-1).to(output.dtype)

    def append_memory(self, x: Tensor, cache: KVCache, position: float) -> None:
        key, value = self._key_value(x)
        positions = torch.full((x.shape[0], 1), position, device=x.device, dtype=torch.float32)
        cos, sin = self.rope(positions, key.dtype)
        cache.append(_apply_rope(key, cos, sin), value)

    def query_cache(self, x: Tensor, cache: KVCache, position: float) -> Tensor:
        query = self._query(x)
        positions = torch.full((x.shape[0], 1), position, device=x.device, dtype=torch.float32)
        cos, sin = self.rope(positions, query.dtype)
        query = _apply_rope(query, cos, sin)
        key, value = cache.ordered(query.dtype)
        output = F.scaled_dot_product_attention(
            query,
            self._repeat_kv(key),
            self._repeat_kv(value),
            dropout_p=0.0,
            is_causal=False,
        )
        output = output.transpose(1, 2).contiguous().view(x.shape[0], 1, self.d_model)
        return self.out_proj(output)

    def incremental(self, x: Tensor, cache: KVCache, *, position: float) -> Tensor:
        self.append_memory(x, cache, position)
        return self.query_cache(x, cache, position)


class LocalTransformerBlock(nn.Module):
    def __init__(self, config: CARSRConfig, *, local_window: int) -> None:
        super().__init__()
        self.local_window = local_window
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = GQAttention(config.d_model, config.n_heads, config.n_kv_heads, config.dropout)
        self.ff_norm = RMSNorm(config.d_model)
        self.ff = SwiGLU(config.d_model, config.d_ff, config.dropout)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, mask: Tensor, positions: Tensor) -> Tensor:
        x = x + self.dropout(
            self.attn(self.attn_norm(x), mask, positions, local_window=self.local_window)
        )
        x = x + self.dropout(self.ff(self.ff_norm(x)))
        return x * mask.unsqueeze(-1).to(x.dtype)

    def incremental(self, x: Tensor, cache: KVCache, *, position: int) -> Tensor:
        x = x + self.attn.incremental(self.attn_norm(x), cache, position=float(position))
        return x + self.ff(self.ff_norm(x))


# -----------------------------------------------------------------------------
# Patching and span geometry
# -----------------------------------------------------------------------------


@dataclass
class PatchSpan:
    indices: Tensor
    starts: Tensor
    ends: Tensor
    centres: Tensor
    lengths: Tensor


@dataclass
class PatchOutput:
    patches: Tensor
    patch_mask: Tensor
    token_to_patch: Tensor
    hard_boundaries: Tensor
    boundary_probs: Tensor
    span: PatchSpan
    compression_loss: Tensor

    @property
    def patch_lengths(self) -> Tensor:
        return self.span.lengths

    @property
    def patch_count(self) -> Tensor:
        return self.patch_mask.sum(-1)


class CausalPatchRouter(nn.Module):
    """Causal information-mass router and order-aware patch compressor.

    The learned scalar at each byte is interpreted as information mass, not a
    categorical boundary probability. A patch closes when accumulated mass
    crosses an integer threshold. Hard segmentation is used in the forward
    path; a soft assignment supplies gradients to the router.

    The patch representation concatenates:

    * the causal mean state;
    * the ending state;
    * an ordinal-weighted state that changes when byte order changes;
    * a learned length embedding.
    """

    def __init__(self, config: CARSRConfig) -> None:
        super().__init__()
        self.config = config
        d = config.d_model
        self.router = nn.Sequential(
            RMSNorm(2 * d + 1),
            nn.Linear(2 * d + 1, d, bias=False),
            nn.SiLU(),
            nn.Linear(d, 1),
        )
        self.length_embedding = nn.Embedding(config.max_patch_size + 1, d)
        self.compressor = nn.Linear(4 * d, d, bias=False)
        self.reset_router_bias()

    def reset_router_bias(self) -> None:
        minimum = 1.0 / self.config.max_patch_size
        maximum = 1.0 / self.config.min_patch_size
        target = 1.0 / self.config.target_patch_ratio
        initial = (target - minimum) / max(maximum - minimum, 1e-8)
        initial = min(max(initial, 1e-4), 1.0 - 1e-4)
        nn.init.constant_(self.router[-1].bias, math.log(initial / (1.0 - initial)))

    def _fixed_boundaries(self, mask: Tensor) -> Tensor:
        positions = torch.arange(mask.shape[1], device=mask.device)[None]
        if self.config.patch_mode == "none":
            return mask.clone()
        return ((positions + 1) % self.config.target_patch_ratio == 0) & mask

    def _learned_boundaries(
        self, hidden: Tensor, mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        previous = torch.cat((torch.zeros_like(hidden[:, :1]), hidden[:, :-1]), dim=1)
        delta = hidden - previous
        cosine = F.cosine_similarity(hidden.float(), previous.float(), dim=-1).to(hidden.dtype)
        raw = torch.sigmoid(
            self.router(torch.cat((hidden, delta, cosine.unsqueeze(-1)), dim=-1)).squeeze(-1)
        )
        minimum = 1.0 / self.config.max_patch_size
        maximum = 1.0 / self.config.min_patch_size
        information_mass = (minimum + (maximum - minimum) * raw) * mask.to(raw.dtype)

        # Reference bounded reset scan. It stays entirely on device, guarantees
        # min/max patch lengths, and preserves differentiable within-patch phase.
        batch, time = mask.shape
        accumulator = hidden.new_zeros(batch)
        current_length = torch.zeros(batch, device=hidden.device, dtype=torch.long)
        patch_id = torch.zeros(batch, device=hidden.device, dtype=torch.long)
        boundaries: list[Tensor] = []
        hard_ids: list[Tensor] = []
        soft_ids: list[Tensor] = []
        for index in range(time):
            valid = mask[:, index]
            hard_ids.append(patch_id)
            soft_ids.append(patch_id.to(hidden.dtype) + accumulator)
            proposed = accumulator + information_mass[:, index]
            length = current_length + valid.long()
            eligible = length >= self.config.min_patch_size
            forced = length >= self.config.max_patch_size
            boundary = valid & (forced | (eligible & proposed.ge(1.0)))
            boundaries.append(boundary)
            patch_id = patch_id + boundary.long()
            accumulator = torch.where(boundary, torch.zeros_like(proposed), proposed)
            current_length = torch.where(boundary, torch.zeros_like(length), length)
        return (
            torch.stack(boundaries, dim=1),
            information_mass,
            torch.stack(hard_ids, dim=1),
            torch.stack(soft_ids, dim=1),
        )

    def _span_geometry(
        self,
        hard_assignment: Tensor,
        mask: Tensor,
        patch_mask: Tensor,
    ) -> PatchSpan:
        batch, time, patches = hard_assignment.shape
        positions = torch.arange(time, device=mask.device, dtype=torch.long)[None, :, None]
        assigned = hard_assignment.bool() & mask[:, :, None]
        starts = torch.where(assigned, positions, time).amin(1)
        ends = torch.where(assigned, positions, -1).amax(1)
        lengths = assigned.sum(1).long()
        starts = torch.where(patch_mask, starts, torch.zeros_like(starts))
        ends = torch.where(patch_mask, ends, torch.zeros_like(ends))
        centres = (starts.to(torch.float32) + ends.to(torch.float32)) * 0.5
        indices = torch.arange(patches, device=mask.device)[None].expand(batch, -1)
        return PatchSpan(indices=indices, starts=starts, ends=ends, centres=centres, lengths=lengths)

    def forward(self, hidden: Tensor, mask: Tensor) -> PatchOutput:
        if self.config.patch_mode == "learned":
            boundaries, mass, hard_ids, soft_ids = self._learned_boundaries(hidden, mask)
        else:
            boundaries = self._fixed_boundaries(mask)
            mass = boundaries.to(hidden.dtype)
            hard_ids = boundaries.long().cumsum(1) - boundaries.long()
            soft_ids = hard_ids.to(hidden.dtype)

        max_patches = min(self.config.max_patches, hidden.shape[1])
        hard_ids = hard_ids.clamp(0, max_patches - 1)
        centres = torch.arange(max_patches, device=hidden.device, dtype=hidden.dtype)
        distances = (soft_ids.unsqueeze(-1) - centres).abs()
        soft_assignment = torch.softmax(-distances / self.config.assignment_temperature, dim=-1)
        hard_assignment = F.one_hot(hard_ids, max_patches).to(hidden.dtype)
        assignment = hard_assignment + soft_assignment - soft_assignment.detach()
        assignment = assignment * mask.unsqueeze(-1).to(hidden.dtype)

        counts = assignment.sum(1)
        patch_mask = counts.detach() > 0
        mean = torch.einsum("btp,btd->bpd", assignment, hidden) / counts.clamp_min(1e-6).unsqueeze(-1)

        valid_lengths = mask.sum(-1).clamp_min(1)
        tail = valid_lengths - 1
        representation_ends = boundaries.clone()
        representation_ends.scatter_(1, tail[:, None], True)
        end_weights = assignment * representation_ends.to(hidden.dtype).unsqueeze(-1)
        end_state = torch.einsum("btp,btd->bpd", end_weights, hidden)
        end_state = end_state / end_weights.sum(1).clamp_min(1e-6).unsqueeze(-1)

        token_ordinal = (
            torch.arange(hidden.shape[1], device=hidden.device, dtype=hidden.dtype)[None, :, None]
        )
        span = self._span_geometry(hard_assignment, mask, patch_mask)
        relative_ordinal = token_ordinal - span.starts[:, None].to(hidden.dtype)
        relative_ordinal = (relative_ordinal + 1.0).clamp_min(0.0)
        weighted_assignment = assignment * relative_ordinal
        weighted = torch.einsum("btp,btd->bpd", weighted_assignment, hidden)
        weighted = weighted / weighted_assignment.sum(1).clamp_min(1e-6).unsqueeze(-1)

        length_index = span.lengths.clamp(0, self.config.max_patch_size)
        length_state = self.length_embedding(length_index)
        patches = self.compressor(torch.cat((mean, end_state, weighted, length_state), dim=-1))
        patches = patches * patch_mask.unsqueeze(-1).to(patches.dtype)

        if self.config.patch_mode == "learned":
            valid = mask.sum(-1).clamp_min(1).to(hidden.dtype)
            actual_rate = mass.sum(-1) / valid
            target_rate = 1.0 / self.config.target_patch_ratio
            compression_loss = (actual_rate - target_rate).square().mean()
        else:
            compression_loss = hidden.new_zeros(())

        return PatchOutput(
            patches=patches,
            patch_mask=patch_mask,
            token_to_patch=hard_ids,
            hard_boundaries=boundaries,
            boundary_probs=mass,
            span=span,
            compression_loss=compression_loss,
        )

    def incremental_boundary(
        self,
        hidden: Tensor,
        previous_hidden: Tensor,
        current_length: Tensor,
        accumulator: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        delta = hidden - previous_hidden
        cosine = F.cosine_similarity(hidden.float(), previous_hidden.float(), dim=-1).to(hidden.dtype)
        raw = torch.sigmoid(self.router(torch.cat((hidden, delta, cosine[:, None]), dim=-1)).squeeze(-1))
        if self.config.patch_mode == "none":
            hard = torch.ones_like(raw, dtype=torch.bool)
            mass = torch.ones_like(raw)
            next_accumulator = torch.zeros_like(accumulator)
        elif self.config.patch_mode == "fixed":
            hard = current_length + 1 >= self.config.target_patch_ratio
            mass = hard.to(raw.dtype)
            next_accumulator = torch.zeros_like(accumulator)
        else:
            minimum = 1.0 / self.config.max_patch_size
            maximum = 1.0 / self.config.min_patch_size
            mass = minimum + (maximum - minimum) * raw
            proposed = accumulator + mass
            next_length = current_length + 1
            eligible = next_length >= self.config.min_patch_size
            forced = next_length >= self.config.max_patch_size
            hard = forced | (eligible & proposed.ge(1.0))
            next_accumulator = torch.where(hard, torch.zeros_like(proposed), proposed)
        return hard, mass, next_accumulator

    def compress_incremental(
        self,
        patch_sum: Tensor,
        weighted_sum: Tensor,
        patch_end: Tensor,
        length: Tensor,
    ) -> Tensor:
        length_float = length.clamp_min(1).to(patch_sum.dtype).unsqueeze(-1)
        mean = patch_sum / length_float
        triangular = (length_float * (length_float + 1.0) * 0.5).clamp_min(1.0)
        weighted = weighted_sum / triangular
        length_state = self.length_embedding(length.clamp(0, self.config.max_patch_size))
        return self.compressor(torch.cat((mean, patch_end, weighted, length_state), dim=-1)).unsqueeze(1)


# -----------------------------------------------------------------------------
# Patch-aware global attention
# -----------------------------------------------------------------------------


class GlobalAttention(Protocol):
    def allocate_cache(self, batch: int, capacity: int, device: torch.device, dtype: torch.dtype) -> AttentionCache: ...


class CPLA(nn.Module):
    """CARS-R Patch Latent Attention.

    CPLA compresses each completed patch into one shared content latent and one
    decoupled span-position key. Queries remain head-specific. Values are
    reconstructed from the shared latent only when used, so cache size scales
    with ``content_dim + position_dim`` rather than full per-head K and V.
    """

    def __init__(self, config: CARSRConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.content_dim = config.cpla_content_dim
        self.position_dim = config.cpla_position_dim
        self.q_content = nn.Linear(config.d_model, config.n_heads * self.content_dim, bias=False)
        self.kv_down = nn.Linear(config.d_model, self.content_dim, bias=False)
        self.q_position = nn.Linear(config.d_model, config.n_heads * self.position_dim, bias=False)
        self.k_position = nn.Linear(config.d_model, self.position_dim, bias=False)
        self.value_up = nn.Linear(self.content_dim, config.n_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.span_rope = DualSpanRoPE(self.position_dim)
        self.dropout = config.dropout
        self.span_mass = nn.Parameter(torch.zeros(config.n_heads)) if config.span_mass_bias else None

    def allocate_cache(
        self,
        batch: int,
        capacity: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> CPLACache:
        return CPLACache.allocate(
            batch=batch,
            capacity=capacity,
            content_dim=self.content_dim,
            position_dim=self.position_dim,
            device=device,
            dtype=dtype,
        )

    def _query(self, x: Tensor, span: PatchSpan) -> tuple[Tensor, Tensor]:
        batch, time, _ = x.shape
        content = self.q_content(x).view(batch, time, self.n_heads, self.content_dim).transpose(1, 2)
        position = self.q_position(x).view(batch, time, self.n_heads, self.position_dim).transpose(1, 2)
        position = self.span_rope(position, span.indices, span.centres)
        return content, position

    def _memory(self, x: Tensor, span: PatchSpan) -> tuple[Tensor, Tensor]:
        latent = self.kv_down(x)
        position = self.k_position(x).unsqueeze(1)
        position = self.span_rope(position, span.indices, span.centres).squeeze(1)
        return latent, position

    def _attend(
        self,
        query_content: Tensor,
        query_position: Tensor,
        latent: Tensor,
        position_key: Tensor,
        query_mask: Tensor,
        memory_mask: Tensor,
        query_indices: Tensor,
        memory_indices: Tensor,
        memory_lengths: Tensor,
    ) -> Tensor:
        content_score = torch.einsum("bhtc,bsc->bhts", query_content, latent)
        position_score = torch.einsum("bhtd,bsd->bhts", query_position, position_key)
        scores = (content_score + position_score) / math.sqrt(self.content_dim + self.position_dim)
        if self.span_mass is not None:
            scores = scores + self.span_mass[None, :, None, None] * torch.log1p(
                memory_lengths.to(scores.dtype)
            )[:, None, None, :]
        allowed = memory_indices[:, None, :] <= query_indices[:, :, None]
        allowed &= memory_mask[:, None, :]
        safe = torch.where(
            query_mask[:, :, None],
            allowed,
            F.one_hot(
                torch.zeros(query_mask.shape, device=query_mask.device, dtype=torch.long),
                memory_mask.shape[1],
            ).bool(),
        )
        scores = scores.masked_fill(~safe[:, None], torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        value = self.value_up(latent).view(
            latent.shape[0], latent.shape[1], self.n_heads, self.head_dim
        ).transpose(1, 2)
        output = torch.einsum("bhts,bhsd->bhtd", weights, value)
        output = output.transpose(1, 2).contiguous().view(
            query_content.shape[0], query_content.shape[2], self.d_model
        )
        return self.out_proj(output) * query_mask.unsqueeze(-1).to(output.dtype)

    def forward_self(self, x: Tensor, mask: Tensor, span: PatchSpan) -> Tensor:
        q_content, q_position = self._query(x, span)
        latent, position_key = self._memory(x, span)
        return self._attend(
            q_content,
            q_position,
            latent,
            position_key,
            mask,
            mask,
            span.indices,
            span.indices,
            span.lengths,
        )

    def forward_cross(
        self,
        query_x: Tensor,
        memory_x: Tensor,
        query_mask: Tensor,
        memory_mask: Tensor,
        query_span: PatchSpan,
        memory_span: PatchSpan,
    ) -> Tensor:
        q_content, q_position = self._query(query_x, query_span)
        latent, position_key = self._memory(memory_x, memory_span)
        return self._attend(
            q_content,
            q_position,
            latent,
            position_key,
            query_mask,
            memory_mask,
            query_span.indices,
            memory_span.indices,
            memory_span.lengths,
        )

    def append_memory(self, x: Tensor, cache: CPLACache, span: PatchSpan) -> None:
        latent, position_key = self._memory(x, span)
        cache.append(latent, position_key, span.lengths[:, 0])

    def query_cache(self, x: Tensor, cache: CPLACache, span: PatchSpan) -> Tensor:
        q_content, q_position = self._query(x, span)
        latent, position_key, lengths = cache.ordered(q_content.dtype)
        memory_count = latent.shape[1]
        query_mask = torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)
        memory_mask = torch.ones(x.shape[0], memory_count, device=x.device, dtype=torch.bool)
        memory_indices = torch.arange(memory_count, device=x.device)[None].expand(x.shape[0], -1)
        return self._attend(
            q_content,
            q_position,
            latent,
            position_key,
            query_mask,
            memory_mask,
            span.indices,
            memory_indices,
            lengths,
        )

    def incremental_self(self, x: Tensor, cache: CPLACache, span: PatchSpan) -> Tensor:
        self.append_memory(x, cache, span)
        return self.query_cache(x, cache, span)


class PatchGQA(nn.Module):
    """Matched global GQA control using byte-centre positions."""

    def __init__(self, config: CARSRConfig) -> None:
        super().__init__()
        self.attention = GQAttention(
            config.d_model, config.n_heads, config.n_kv_heads, config.dropout
        )
        self.config = config

    def allocate_cache(
        self,
        batch: int,
        capacity: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> KVCache:
        return KVCache.allocate(
            batch=batch,
            n_kv_heads=self.config.n_kv_heads,
            capacity=capacity,
            head_dim=self.config.d_model // self.config.n_heads,
            device=device,
            dtype=dtype,
        )

    def forward_self(self, x: Tensor, mask: Tensor, span: PatchSpan) -> Tensor:
        return self.attention(x, mask, span.centres, local_window=None)

    def forward_cross(
        self,
        query_x: Tensor,
        memory_x: Tensor,
        query_mask: Tensor,
        memory_mask: Tensor,
        query_span: PatchSpan,
        memory_span: PatchSpan,
    ) -> Tensor:
        return self.attention.forward_cross(
            query_x,
            memory_x,
            query_mask,
            memory_mask,
            query_span.centres,
            memory_span.centres,
        )

    def append_memory(self, x: Tensor, cache: KVCache, span: PatchSpan) -> None:
        self.attention.append_memory(x, cache, float(span.centres[0, 0].item()))

    def query_cache(self, x: Tensor, cache: KVCache, span: PatchSpan) -> Tensor:
        return self.attention.query_cache(x, cache, float(span.centres[0, 0].item()))

    def incremental_self(self, x: Tensor, cache: KVCache, span: PatchSpan) -> Tensor:
        self.append_memory(x, cache, span)
        return self.query_cache(x, cache, span)


def make_patch_attention(config: CARSRConfig) -> CPLA | PatchGQA:
    return CPLA(config) if config.patch_attention == "cpla" else PatchGQA(config)


class PatchTransformerBlock(nn.Module):
    def __init__(self, config: CARSRConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = make_patch_attention(config)
        self.ff_norm = RMSNorm(config.d_model)
        self.ff = SwiGLU(config.d_model, config.d_ff, config.dropout)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor, mask: Tensor, span: PatchSpan) -> Tensor:
        x = x + self.dropout(self.attn.forward_self(self.attn_norm(x), mask, span))
        x = x + self.dropout(self.ff(self.ff_norm(x)))
        return x * mask.unsqueeze(-1).to(x.dtype)

    def incremental(self, x: Tensor, cache: AttentionCache, span: PatchSpan) -> Tensor:
        x = x + self.attn.incremental_self(self.attn_norm(x), cache, span)
        return x + self.ff(self.ff_norm(x))

    def allocate_cache(
        self, batch: int, capacity: int, device: torch.device, dtype: torch.dtype
    ) -> AttentionCache:
        return self.attn.allocate_cache(batch, capacity, device, dtype)


class RecurrentPatchBlock(nn.Module):
    """Shared recurrent update over one fixed historical patch memory."""

    def __init__(self, config: CARSRConfig) -> None:
        super().__init__()
        self.query_norm = RMSNorm(config.d_model)
        self.memory_norm = RMSNorm(config.d_model)
        self.attn = make_patch_attention(config)
        self.ff_norm = RMSNorm(config.d_model)
        self.ff = SwiGLU(config.d_model, config.d_ff, config.dropout)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        query: Tensor,
        memory: Tensor,
        mask: Tensor,
        span: PatchSpan,
    ) -> Tensor:
        attention = self.attn.forward_cross(
            self.query_norm(query),
            self.memory_norm(memory),
            mask,
            mask,
            span,
            span,
        )
        intermediate = query + self.dropout(attention)
        return attention + self.dropout(self.ff(self.ff_norm(intermediate)))

    def append_memory(self, memory: Tensor, cache: AttentionCache, span: PatchSpan) -> None:
        self.attn.append_memory(self.memory_norm(memory), cache, span)

    def incremental_update(
        self,
        query: Tensor,
        cache: AttentionCache,
        span: PatchSpan,
    ) -> Tensor:
        attention = self.attn.query_cache(self.query_norm(query), cache, span)
        intermediate = query + attention
        return attention + self.ff(self.ff_norm(intermediate))

    def allocate_cache(
        self, batch: int, capacity: int, device: torch.device, dtype: torch.dtype
    ) -> AttentionCache:
        return self.attn.allocate_cache(batch, capacity, device, dtype)


# -----------------------------------------------------------------------------
# Complete model
# -----------------------------------------------------------------------------


@dataclass
class CARSRModelOutput:
    logits: Tensor
    loss: Tensor | None
    token_loss: Tensor | None
    compression_loss: Tensor
    patching: PatchOutput
    recurrent_depth: int
    recurrence_update_norms: Tensor


class CARSRModel(nn.Module):
    """CARS-R 0.1.0: causal adaptive recurrent scaling over raw bytes."""

    checkpoint_version = "0.1.0"

    def __init__(self, config: CARSRConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.byte_encoder = nn.ModuleList(
            LocalTransformerBlock(config, local_window=config.byte_window)
            for _ in range(config.byte_layers)
        )
        self.patcher = CausalPatchRouter(config)
        self.patch_blocks = nn.ModuleList(PatchTransformerBlock(config) for _ in range(config.patch_layers))
        self.recurrent_block = RecurrentPatchBlock(config)
        max_depth = max(config.recurrent_depths, default=0)
        self.iteration_embedding = nn.Embedding(max_depth + 1, config.d_model)
        self.recurrent_step_logits = nn.Parameter(torch.zeros(max_depth + 1))
        self.patch_norm = RMSNorm(config.d_model)
        self.context_projection = nn.Linear(config.d_model, config.d_model, bias=False)
        self.context_gate = nn.Linear(2 * config.d_model + 3, config.d_model)
        self.byte_decoder = nn.ModuleList(
            LocalTransformerBlock(config, local_window=config.byte_window)
            for _ in range(config.decoder_layers)
        )
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)
        self.patcher.reset_router_bias()
        nn.init.constant_(self.context_gate.bias, -1.0)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def _validate_inputs(self, input_ids: Tensor, attention_mask: Tensor | None) -> Tensor:
        if input_ids.ndim != 2 or input_ids.shape[1] < 1:
            raise ValueError("input_ids must be [batch,time]")
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("sequence exceeds max_seq_len")
        if input_ids.dtype not in {torch.int32, torch.int64}:
            raise TypeError("input_ids must be integer")
        if bool((input_ids < 0).any() or (input_ids >= self.config.vocab_size).any()):
            raise ValueError("input token outside vocabulary")
        mask = input_ids.ne(self.config.pad_token_id) if attention_mask is None else attention_mask.bool()
        if mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        if mask.shape[1] > 1 and bool((mask[:, 1:] & ~mask[:, :-1]).any()):
            raise ValueError("attention_mask must be a contiguous valid prefix")
        return mask

    def _run_patch_core(
        self,
        patches: Tensor,
        mask: Tensor,
        span: PatchSpan,
        depth: int,
    ) -> tuple[Tensor, Tensor]:
        base = patches
        for block in self.patch_blocks:
            base = block(base, mask, span)
        x = base
        norms: list[Tensor] = []
        for iteration in range(depth):
            query = x + self.iteration_embedding.weight[iteration + 1]
            update = self.recurrent_block(query, base, mask, span)
            step = 2.0 * torch.sigmoid(self.recurrent_step_logits[iteration + 1])
            step = step / math.sqrt(iteration + 1.0)
            x = x + step * update
            norms.append(update.detach().float().norm(dim=-1)[mask].mean())
        recorded = torch.stack(norms) if norms else patches.new_empty(0, dtype=torch.float32)
        return self.patch_norm(x), recorded

    def _expand_completed_patch_context(
        self,
        patch_states: Tensor,
        patching: PatchOutput,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        completed = patching.token_to_patch - (~patching.hard_boundaries).long()
        valid = completed.ge(0) & mask
        indices = completed.clamp_min(0).clamp_max(patch_states.shape[1] - 1)
        context = patch_states.gather(
            1, indices.unsqueeze(-1).expand(-1, -1, patch_states.shape[-1])
        )
        completed_ends = patching.span.ends.gather(1, indices)
        completed_lengths = patching.span.lengths.gather(1, indices)
        token_positions = torch.arange(mask.shape[1], device=mask.device)[None]
        age = (token_positions - completed_ends).clamp_min(0)
        context = context * valid.unsqueeze(-1).to(context.dtype)
        age = age * valid
        completed_lengths = completed_lengths * valid
        return context, age, completed_lengths

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor | None = None,
        *,
        labels: Tensor | None = None,
        recurrent_depth: int | None = None,
    ) -> CARSRModelOutput:
        mask = self._validate_inputs(input_ids, attention_mask)
        depth = self.config.default_recurrent_depth if recurrent_depth is None else recurrent_depth
        if depth not in self.config.recurrent_depths:
            raise ValueError("recurrent_depth must be one of config.recurrent_depths")
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        local = self.token_embedding(input_ids)
        for block in self.byte_encoder:
            local = block(local, mask, positions)

        patching = self.patcher(local, mask)
        patch_states, update_norms = self._run_patch_core(
            patching.patches, patching.patch_mask, patching.span, depth
        )
        raw_context, context_age, context_length = self._expand_completed_patch_context(
            patch_states, patching, mask
        )
        context = self.context_projection(raw_context)
        age_feature = torch.log1p(context_age.to(local.dtype)).unsqueeze(-1)
        length_feature = torch.log1p(context_length.to(local.dtype)).unsqueeze(-1)
        boundary_feature = patching.boundary_probs.unsqueeze(-1).to(local.dtype)
        gate = torch.sigmoid(
            self.context_gate(
                torch.cat((local, context, age_feature, length_feature, boundary_feature), dim=-1)
            )
        )
        decoded = local + gate * context
        for block in self.byte_decoder:
            decoded = block(decoded, mask, positions)
        logits = self.lm_head(self.final_norm(decoded))

        token_loss: Tensor | None = None
        loss: Tensor | None = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must match input_ids")
            token_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=self.config.pad_token_id,
            )
            loss = token_loss + self.config.compression_loss_weight * patching.compression_loss

        return CARSRModelOutput(
            logits=logits,
            loss=loss,
            token_loss=token_loss,
            compression_loss=patching.compression_loss,
            patching=patching,
            recurrent_depth=depth,
            recurrence_update_norms=update_norms,
        )

    def _cache_dtype(self) -> torch.dtype:
        if self.config.cache_dtype == "float16":
            return torch.float16
        if self.config.cache_dtype == "bfloat16":
            return torch.bfloat16
        return self.token_embedding.weight.dtype

    def _allocate_local_caches(self, blocks: nn.ModuleList, batch: int) -> list[KVCache]:
        device = self.token_embedding.weight.device
        dtype = self._cache_dtype()
        return [
            KVCache.allocate(
                batch=batch,
                n_kv_heads=self.config.n_kv_heads,
                capacity=self.config.byte_window,
                head_dim=self.config.d_model // self.config.n_heads,
                device=device,
                dtype=dtype,
            )
            for _ in blocks
        ]

    def _allocate_patch_caches(self, batch: int) -> list[AttentionCache]:
        device = self.token_embedding.weight.device
        dtype = self._cache_dtype()
        return [
            block.allocate_cache(batch, self.config.max_patches, device, dtype)
            for block in self.patch_blocks
        ]

    @torch.inference_mode()
    def start_generation_session(
        self,
        input_ids: Tensor,
        *,
        recurrent_depth: int | None = None,
    ) -> "GenerationSession":
        if self.training:
            raise RuntimeError("generation sessions require model.eval()")
        self._validate_inputs(input_ids, None)
        if bool(input_ids.eq(self.config.pad_token_id).any()):
            raise ValueError("generation prompts cannot contain PAD tokens")
        if input_ids.shape[0] != 1:
            raise ValueError("research generation sessions currently support batch size one")
        depth = self.config.default_recurrent_depth if recurrent_depth is None else recurrent_depth
        if depth not in self.config.recurrent_depths:
            raise ValueError("invalid recurrent depth")
        state = GenerationState.create(self, input_ids.shape[0], depth)
        logits = None
        for index in range(input_ids.shape[1]):
            logits = self._increment_token(input_ids[:, index : index + 1], state)
        assert logits is not None
        return GenerationSession(self, state, input_ids.clone(), logits)

    def _single_span(
        self,
        batch: int,
        patch_index: int,
        start: int,
        end: int,
        device: torch.device,
    ) -> PatchSpan:
        length = end - start + 1
        return PatchSpan(
            indices=torch.full((batch, 1), patch_index, device=device, dtype=torch.long),
            starts=torch.full((batch, 1), start, device=device, dtype=torch.long),
            ends=torch.full((batch, 1), end, device=device, dtype=torch.long),
            centres=torch.full((batch, 1), (start + end) * 0.5, device=device),
            lengths=torch.full((batch, 1), length, device=device, dtype=torch.long),
        )

    def _increment_patch(
        self,
        patch: Tensor,
        state: "GenerationState",
        span: PatchSpan,
    ) -> Tensor:
        base = patch
        for block, cache in zip(self.patch_blocks, state.patch_caches, strict=True):
            base = block.incremental(base, cache, span)
        self.recurrent_block.append_memory(base, state.recurrent_cache, span)
        x = base
        for iteration in range(state.recurrent_depth):
            query = x + self.iteration_embedding.weight[iteration + 1]
            update = self.recurrent_block.incremental_update(query, state.recurrent_cache, span)
            step = 2.0 * torch.sigmoid(self.recurrent_step_logits[iteration + 1])
            step = step / math.sqrt(iteration + 1.0)
            x = x + step * update
        state.patch_count += 1
        return self.patch_norm(x).squeeze(1)

    def _increment_token(self, token_ids: Tensor, state: "GenerationState") -> Tensor:
        if state.byte_position >= self.config.max_seq_len:
            raise RuntimeError("generation cache reached max_seq_len")
        local = self.token_embedding(token_ids)
        for block, cache in zip(self.byte_encoder, state.byte_caches, strict=True):
            local = block.incremental(local, cache, position=state.byte_position)
        local_row = local.squeeze(1)
        boundary, mass, next_accumulator = self.patcher.incremental_boundary(
            local_row,
            state.previous_local,
            state.current_patch_length,
            state.boundary_accumulator,
        )

        next_length = state.current_patch_length + 1
        state.patch_sum = state.patch_sum + local_row
        state.patch_weighted_sum = state.patch_weighted_sum + next_length.to(local_row.dtype).unsqueeze(-1) * local_row
        state.current_patch_length = next_length
        completed_context = state.last_patch_state
        completed_end = state.last_patch_end
        completed_length = state.last_patch_length

        if bool(boundary.any().item()) and not bool(boundary.all().item()):
            raise RuntimeError("batched generation currently requires aligned patch boundaries")
        if bool(boundary.all().item()):
            patch = self.patcher.compress_incremental(
                state.patch_sum,
                state.patch_weighted_sum,
                local_row,
                state.current_patch_length,
            )
            start = state.byte_position - int(state.current_patch_length[0].item()) + 1
            span = self._single_span(
                token_ids.shape[0], state.patch_count, start, state.byte_position, token_ids.device
            )
            completed_context = self._increment_patch(patch, state, span)
            state.last_patch_state = completed_context
            state.last_patch_end.fill_(state.byte_position)
            state.last_patch_length.copy_(state.current_patch_length)
            completed_end = state.last_patch_end
            completed_length = state.last_patch_length
            state.patch_sum.zero_()
            state.patch_weighted_sum.zero_()
            state.current_patch_length.zero_()

        projected = self.context_projection(completed_context).unsqueeze(1)
        valid_context = completed_end.ge(0)
        age = (state.byte_position - completed_end).clamp_min(0)
        features = torch.cat(
            (
                local,
                projected,
                torch.log1p(age.to(local.dtype))[:, None, None],
                torch.log1p(completed_length.to(local.dtype))[:, None, None],
                mass[:, None, None],
            ),
            dim=-1,
        )
        gate = torch.sigmoid(self.context_gate(features))
        decoded = local + gate * projected * valid_context[:, None, None].to(local.dtype)
        for block, cache in zip(self.byte_decoder, state.decoder_caches, strict=True):
            decoded = block.incremental(decoded, cache, position=state.byte_position)

        state.previous_local = local_row
        state.boundary_accumulator = next_accumulator
        state.last_boundary_probability = mass
        state.byte_position += 1
        return self.lm_head(self.final_norm(decoded[:, -1]))

    @torch.inference_mode()
    def generate(
        self,
        input_ids: Tensor,
        *,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_k: int | None = None,
        recurrent_depth: int | None = None,
    ) -> Tensor:
        was_training = self.training
        self.eval()
        try:
            session = self.start_generation_session(input_ids, recurrent_depth=recurrent_depth)
            generated = input_ids
            for _ in range(max_new_tokens):
                next_token = session.sample(temperature=temperature, top_k=top_k)
                generated = torch.cat((generated, next_token), dim=1)
                if bool(next_token.eq(self.config.eos_token_id).all().item()):
                    break
                session.append(next_token)
            return generated
        finally:
            self.train(was_training)


@dataclass
class GenerationState:
    byte_caches: list[KVCache]
    patch_caches: list[AttentionCache]
    recurrent_cache: AttentionCache
    decoder_caches: list[KVCache]
    previous_local: Tensor
    patch_sum: Tensor
    patch_weighted_sum: Tensor
    current_patch_length: Tensor
    last_patch_state: Tensor
    last_patch_end: Tensor
    last_patch_length: Tensor
    last_boundary_probability: Tensor
    boundary_accumulator: Tensor
    byte_position: int
    patch_count: int
    recurrent_depth: int

    @classmethod
    def create(cls, model: CARSRModel, batch: int, depth: int) -> "GenerationState":
        device = model.token_embedding.weight.device
        dtype = model.token_embedding.weight.dtype
        cache_dtype = model._cache_dtype()
        return cls(
            byte_caches=model._allocate_local_caches(model.byte_encoder, batch),
            patch_caches=model._allocate_patch_caches(batch),
            recurrent_cache=model.recurrent_block.allocate_cache(
                batch, model.config.max_patches, device, cache_dtype
            ),
            decoder_caches=model._allocate_local_caches(model.byte_decoder, batch),
            previous_local=torch.zeros(batch, model.config.d_model, device=device, dtype=dtype),
            patch_sum=torch.zeros(batch, model.config.d_model, device=device, dtype=dtype),
            patch_weighted_sum=torch.zeros(batch, model.config.d_model, device=device, dtype=dtype),
            current_patch_length=torch.zeros(batch, device=device, dtype=torch.long),
            last_patch_state=torch.zeros(batch, model.config.d_model, device=device, dtype=dtype),
            last_patch_end=torch.full((batch,), -1, device=device, dtype=torch.long),
            last_patch_length=torch.zeros(batch, device=device, dtype=torch.long),
            last_boundary_probability=torch.zeros(batch, device=device, dtype=dtype),
            boundary_accumulator=torch.zeros(batch, device=device, dtype=dtype),
            byte_position=0,
            patch_count=0,
            recurrent_depth=depth,
        )

    @property
    def cache_bytes(self) -> int:
        caches = [*self.byte_caches, *self.patch_caches, self.recurrent_cache, *self.decoder_caches]
        tensors = (
            self.previous_local,
            self.patch_sum,
            self.patch_weighted_sum,
            self.current_patch_length,
            self.last_patch_state,
            self.last_patch_end,
            self.last_patch_length,
            self.last_boundary_probability,
            self.boundary_accumulator,
        )
        return sum(cache.bytes for cache in caches) + sum(
            tensor.numel() * tensor.element_size() for tensor in tensors
        )


@dataclass
class GenerationSession:
    model: CARSRModel
    state: GenerationState
    tokens: Tensor
    logits: Tensor

    @torch.inference_mode()
    def append(self, token_ids: Tensor) -> Tensor:
        if token_ids.ndim == 1:
            token_ids = token_ids[:, None]
        if token_ids.shape != (self.tokens.shape[0], 1):
            raise ValueError("token_ids must be [batch,1]")
        if bool(
            token_ids.eq(self.model.config.pad_token_id).any()
            or token_ids.eq(self.model.config.bos_token_id).any()
        ):
            raise ValueError("generation append rejects PAD and BOS tokens")
        self.model._validate_inputs(token_ids, None)
        self.logits = self.model._increment_token(token_ids, self.state)
        self.tokens = torch.cat((self.tokens, token_ids), dim=1)
        return self.logits

    def sample(self, *, temperature: float = 0.0, top_k: int | None = None) -> Tensor:
        logits = self.logits.clone()
        logits[:, [self.model.config.pad_token_id, self.model.config.bos_token_id]] = -torch.inf
        if temperature == 0.0:
            return logits.argmax(-1, keepdim=True)
        if temperature < 0:
            raise ValueError("temperature cannot be negative")
        logits = logits / temperature
        if top_k:
            threshold = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < threshold, -torch.inf)
        return torch.multinomial(logits.softmax(-1), 1)


# -----------------------------------------------------------------------------
# Checkpoints
# -----------------------------------------------------------------------------


def save_checkpoint(
    path: str | Path,
    model: CARSRModel,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
    metrics: dict[str, float] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "version": CARSRModel.checkpoint_version,
        "config": model.config.to_dict(),
        "model": model.state_dict(),
        "step": step,
        "metrics": metrics or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
    load_optimizer: bool = False,
) -> tuple[CARSRModel, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(payload, dict) or payload.get("version") != CARSRModel.checkpoint_version:
        raise ValueError("checkpoint is not compatible with CARS-R 0.1.0; retraining is required")
    config = CARSRConfig.from_dict(payload["config"])
    model = CARSRModel(config).to(device)
    model.load_state_dict(payload["model"], strict=True)
    metadata: dict[str, Any] = {
        "step": int(payload.get("step", 0)),
        "metrics": dict(payload.get("metrics", {})),
    }
    if load_optimizer and "optimizer" in payload:
        metadata["optimizer"] = payload["optimizer"]
    return model, metadata

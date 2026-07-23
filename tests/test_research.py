from pathlib import Path

import pytest
import torch

from cars import CARSRConfig, CARSRModel, load_checkpoint, save_checkpoint
from cars.model import CausalPatchRouter, GQAttention, _apply_rope


def tiny_config(**changes) -> CARSRConfig:
    values = dict(
        d_model=48,
        n_heads=4,
        n_kv_heads=2,
        d_ff=96,
        dropout=0.0,
        byte_layers=1,
        byte_window=8,
        patch_layers=1,
        decoder_layers=1,
        max_seq_len=40,
        patch_mode="learned",
        patch_attention="cpla",
        target_patch_ratio=4,
        min_patch_size=2,
        max_patch_size=8,
        cpla_content_dim=16,
        cpla_position_dim=8,
        recurrent_depths=(0, 1, 2),
        default_recurrent_depth=1,
        compression_loss_weight=0.0,
    )
    values.update(changes)
    return CARSRConfig(**values)


def random_batch(batch: int = 2, time: int = 24, seed: int = 3) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randint(0, 240, (batch, time), dtype=torch.long)


def test_true_sliding_window_attention_matches_dense_reference() -> None:
    torch.manual_seed(11)
    attention = GQAttention(32, 4, 2, 0.0).eval()
    hidden = torch.randn(2, 7, 32)
    mask = torch.tensor(
        [
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, False, False],
        ]
    )
    positions = torch.arange(hidden.shape[1])
    with torch.no_grad():
        actual = attention(hidden, mask, positions, local_window=3)
        query, key, value = attention._project(hidden)
        expanded_positions = positions.unsqueeze(0).expand(hidden.shape[0], -1)
        cosine, sine = attention.rope(expanded_positions, query.dtype)
        query = _apply_rope(query, cosine, sine)
        key = attention._repeat_kv(_apply_rope(key, cosine, sine))
        value = attention._repeat_kv(value)
        q_position = expanded_positions[:, :, None]
        k_position = expanded_positions[:, None, :]
        allowed = (k_position <= q_position) & (k_position > q_position - 3)
        allowed &= mask[:, None, :]
        allowed = torch.where(
            (~mask)[:, :, None],
            torch.eye(hidden.shape[1], dtype=torch.bool)[None],
            allowed,
        )
        scores = torch.einsum("bhtd,bhsd->bhts", query, key) * (attention.head_dim**-0.5)
        scores = scores.masked_fill(~allowed[:, None], torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        reference = torch.einsum("bhts,bhsd->bhtd", weights, value)
        reference = reference.transpose(1, 2).contiguous().view_as(hidden)
        reference = attention.out_proj(reference) * mask.unsqueeze(-1)
    torch.testing.assert_close(actual, reference, atol=1e-6, rtol=1e-6)


def test_forward_backward_reaches_information_mass_router_from_language_loss() -> None:
    model = CARSRModel(tiny_config())
    tokens = random_batch()
    labels = torch.roll(tokens, -1, 1)
    labels[:, -1] = model.config.eos_token_id
    output = model(tokens, labels=labels)
    assert output.loss is not None
    output.loss.backward()
    gradient = model.patcher.router[-1].weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert gradient.abs().sum() > 0


def test_forward_is_strictly_causal() -> None:
    model = CARSRModel(tiny_config()).eval()
    tokens = random_batch(batch=1, time=24)
    changed = tokens.clone()
    changed[:, 14:] = torch.randint(0, 240, changed[:, 14:].shape)
    with torch.no_grad():
        original = model(tokens).logits
        altered = model(changed).logits
    torch.testing.assert_close(original[:, :14], altered[:, :14], atol=1e-6, rtol=1e-6)


def test_non_prefix_padding_and_generation_pad_are_rejected() -> None:
    model = CARSRModel(tiny_config()).eval()
    tokens = random_batch(batch=1, time=8)
    mask = torch.tensor([[True, True, False, True, False, False, False, False]])
    with pytest.raises(ValueError, match="contiguous valid prefix"):
        model(tokens, attention_mask=mask)

    padded_prompt = tokens.clone()
    padded_prompt[:, -1] = model.config.pad_token_id
    with pytest.raises(ValueError, match="cannot contain PAD"):
        model.start_generation_session(padded_prompt)

    session = model.start_generation_session(tokens[:, :4])
    with pytest.raises(ValueError, match="rejects PAD and BOS"):
        session.append(torch.tensor([[model.config.bos_token_id]]))


def test_padding_is_finite_and_ignored_by_loss() -> None:
    model = CARSRModel(tiny_config())
    tokens = random_batch(batch=2, time=16)
    tokens[1, 10:] = model.config.pad_token_id
    labels = torch.roll(tokens, -1, 1)
    labels[:, -1] = model.config.pad_token_id
    output = model(tokens, labels=labels)
    assert output.loss is not None and torch.isfinite(output.loss)
    assert torch.isfinite(output.logits).all()


def test_patch_lengths_strictly_respect_constraints_except_open_tail() -> None:
    for seed in range(5):
        model = CARSRModel(tiny_config()).eval()
        tokens = random_batch(batch=2, time=39, seed=seed)
        with torch.no_grad():
            patching = model(tokens).patching
        for row in range(tokens.shape[0]):
            lengths = patching.patch_lengths[row][patching.patch_mask[row]].tolist()
            for length in lengths[:-1]:
                assert model.config.min_patch_size <= length <= model.config.max_patch_size
            assert 1 <= lengths[-1] <= model.config.max_patch_size


def test_span_geometry_matches_patch_lengths() -> None:
    model = CARSRModel(tiny_config(patch_mode="fixed")).eval()
    tokens = random_batch(batch=1, time=17)
    with torch.no_grad():
        patching = model(tokens).patching
    valid = patching.patch_mask[0]
    starts = patching.span.starts[0][valid]
    ends = patching.span.ends[0][valid]
    lengths = patching.span.lengths[0][valid]
    assert torch.equal(ends - starts + 1, lengths)
    torch.testing.assert_close(
        patching.span.centres[0][valid],
        (starts.float() + ends.float()) * 0.5,
    )


def test_order_aware_patch_compressor_changes_under_internal_permutation() -> None:
    config = tiny_config(patch_mode="fixed", target_patch_ratio=4)
    router = CausalPatchRouter(config).eval()
    torch.manual_seed(8)
    hidden = torch.randn(1, 4, config.d_model)
    permuted = hidden.clone()
    permuted[:, [0, 1]] = permuted[:, [1, 0]]
    mask = torch.ones(1, 4, dtype=torch.bool)
    with torch.no_grad():
        first = router(hidden, mask).patches[:, 0]
        second = router(permuted, mask).patches[:, 0]
    assert not torch.equal(first, second)


@pytest.mark.parametrize("attention", ["cpla", "gqa"])
@pytest.mark.parametrize("mode", ["learned", "fixed", "none"])
def test_incremental_cache_matches_full_forward(attention: str, mode: str) -> None:
    model = CARSRModel(tiny_config(patch_mode=mode, patch_attention=attention)).eval()
    tokens = random_batch(batch=1, time=19)
    with torch.no_grad():
        reference = model(tokens, recurrent_depth=2).logits[:, -1]
        session = model.start_generation_session(tokens, recurrent_depth=2)
    torch.testing.assert_close(reference, session.logits, atol=3e-6, rtol=3e-6)


def test_incremental_append_matches_recomputation_across_ring_rotation() -> None:
    model = CARSRModel(tiny_config(patch_mode="fixed")).eval()
    tokens = random_batch(batch=1, time=11)
    session = model.start_generation_session(tokens, recurrent_depth=1)
    for token in [31, 32, 33, 34, 35, 36, 37, 38, 39]:
        next_token = torch.tensor([[token]], dtype=torch.long)
        session.append(next_token)
        tokens = torch.cat((tokens, next_token), dim=1)
        with torch.no_grad():
            reference = model(tokens, recurrent_depth=1).logits[:, -1]
        torch.testing.assert_close(reference, session.logits, atol=3e-6, rtol=3e-6)


def test_hierarchical_cache_has_one_recurrent_memory_not_one_per_depth() -> None:
    model = CARSRModel(tiny_config()).eval()
    session = model.start_generation_session(random_batch(batch=1, time=12), recurrent_depth=2)
    assert all(cache.capacity == model.config.byte_window for cache in session.state.byte_caches)
    assert all(cache.capacity == model.config.byte_window for cache in session.state.decoder_caches)
    assert all(cache.capacity == model.config.max_patches for cache in session.state.patch_caches)
    assert session.state.recurrent_cache.capacity == model.config.max_patches
    assert not hasattr(session.state, "recurrent_caches")
    assert session.state.byte_caches[0].length == model.config.byte_window


def test_cpla_patch_cache_is_smaller_than_matched_gqa_cache() -> None:
    prompt = random_batch(batch=1, time=16)
    cpla = CARSRModel(tiny_config(patch_mode="fixed", patch_attention="cpla")).eval()
    gqa = CARSRModel(tiny_config(patch_mode="fixed", patch_attention="gqa")).eval()
    cpla_state = cpla.start_generation_session(prompt).state
    gqa_state = gqa.start_generation_session(prompt).state
    assert cpla_state.patch_caches[0].bytes < gqa_state.patch_caches[0].bytes
    assert cpla_state.recurrent_cache.bytes < gqa_state.recurrent_cache.bytes


def test_recurrent_depth_reuses_parameters_but_changes_computation() -> None:
    model = CARSRModel(tiny_config()).eval()
    tokens = random_batch(batch=1, time=20)
    parameters = model.parameter_count
    with torch.no_grad():
        depth_zero = model(tokens, recurrent_depth=0)
        depth_two = model(tokens, recurrent_depth=2)
    assert model.parameter_count == parameters
    assert not torch.equal(depth_zero.logits, depth_two.logits)
    assert depth_zero.recurrence_update_norms.numel() == 0
    assert depth_two.recurrence_update_norms.numel() == 2


def test_predetermined_controls_do_not_receive_compression_penalties() -> None:
    tokens = random_batch(batch=1, time=17)
    for mode in ("fixed", "none"):
        model = CARSRModel(tiny_config(patch_mode=mode)).eval()
        with torch.no_grad():
            output = model(tokens)
        assert output.compression_loss.item() == 0.0


def test_fixed_and_none_controls_have_expected_patch_counts() -> None:
    tokens = random_batch(batch=1, time=17)
    fixed = CARSRModel(tiny_config(patch_mode="fixed")).eval()
    none = CARSRModel(tiny_config(patch_mode="none")).eval()
    with torch.no_grad():
        fixed_count = fixed(tokens).patching.patch_count.item()
        none_count = none(tokens).patching.patch_count.item()
    assert fixed_count == 5
    assert none_count == tokens.shape[1]


def test_checkpoint_round_trip_and_incompatible_rejection(tmp_path: Path) -> None:
    model = CARSRModel(tiny_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    path = tmp_path / "research.pt"
    save_checkpoint(path, model, optimizer=optimizer, step=7, metrics={"loss": 1.5})
    loaded, metadata = load_checkpoint(path, load_optimizer=True)
    assert loaded.config.to_dict() == model.config.to_dict()
    assert metadata["step"] == 7
    assert metadata["metrics"] == {"loss": 1.5}
    for expected, actual in zip(model.parameters(), loaded.parameters(), strict=True):
        torch.testing.assert_close(expected, actual)

    incompatible = tmp_path / "incompatible.pt"
    torch.save({"version": "different", "config": {}, "model": {}}, incompatible)
    with pytest.raises(ValueError, match="retraining is required"):
        load_checkpoint(incompatible)


def test_generation_session_rejects_ragged_batched_research_serving() -> None:
    model = CARSRModel(tiny_config()).eval()
    with pytest.raises(ValueError, match="batch size one"):
        model.start_generation_session(random_batch(batch=2, time=8))


def test_normal_inverse_gamma_posterior_updates_and_round_trips() -> None:
    from cars.experiment import NormalInverseGammaPosterior

    posterior = NormalInverseGammaPosterior(
        prior_mean=-8.0,
        prior_strength=0.5,
        prior_alpha=2.0,
        prior_beta=1.0,
    )
    for reward in (-7.0, -6.0, -5.0):
        posterior.update(reward)
    restored = NormalInverseGammaPosterior.from_dict(posterior.to_dict())
    assert restored.count == 3
    assert restored.posterior_mean > -8.0
    assert restored.posterior_std > 0.0
    assert restored.sample(seed=91) == posterior.sample(seed=91)


def test_thompson_scheduler_bootstraps_every_arm_and_persists(tmp_path: Path) -> None:
    from cars.experiment import (
        NormalInverseGammaPosterior,
        ThompsonArm,
        ThompsonResearchScheduler,
    )

    scheduler = ThompsonResearchScheduler(
        [
            ThompsonArm("a", {}, NormalInverseGammaPosterior()),
            ThompsonArm("b", {}, NormalInverseGammaPosterior()),
        ],
        seed=5,
        min_pulls=1,
    )
    first, _ = scheduler.select()
    scheduler.update(first.name, reward=1.0, allocated_steps=10, summary={"x": 1.0})
    second, _ = scheduler.select()
    assert first.name != second.name
    scheduler.update(second.name, reward=0.0, allocated_steps=10, summary={"x": 2.0})

    state = tmp_path / "thompson.json"
    scheduler.save(state)
    restored = ThompsonResearchScheduler.load(state)
    assert restored.round_index == 2
    assert restored.arms["a"].pulls == 1
    assert restored.arms["b"].pulls == 1
    selected, samples = restored.select()
    assert selected.name in {"a", "b"}
    assert all(torch.isfinite(torch.tensor(value)) for value in samples.values())


def test_research_reward_values_cpla_cache_efficiency_without_changing_quality() -> None:
    from cars.experiment import compute_research_reward

    summary = {
        "depth_1/bits_per_byte": 4.0,
        "depth_1/high_risk_accuracy": 0.75,
        "depth_1/mean_patch_length": 4.0,
        "train/tokens_per_second": 100.0,
    }
    cpla = tiny_config(patch_attention="cpla", default_recurrent_depth=1)
    gqa = tiny_config(patch_attention="gqa", default_recurrent_depth=1)
    cpla_reward, cpla_parts = compute_research_reward(
        summary,
        cpla,
        exact_weight=0.05,
        cache_weight=0.2,
        depth_weight=0.0,
        throughput_weight=0.0,
    )
    gqa_reward, gqa_parts = compute_research_reward(
        summary,
        gqa,
        exact_weight=0.05,
        cache_weight=0.2,
        depth_weight=0.0,
        throughput_weight=0.0,
    )
    assert cpla_parts["cache_ratio"] < gqa_parts["cache_ratio"]
    assert cpla_reward > gqa_reward


def test_thompson_command_runs_one_incremental_research_round(tmp_path: Path) -> None:
    from cars.experiment import build_parser

    output = tmp_path / "study"
    args = build_parser().parse_args(
        [
            "thompson",
            "--study",
            "core",
            "--rounds",
            "1",
            "--round-steps",
            "1",
            "--min-pulls",
            "1",
            "--output",
            str(output),
            "--device",
            "cpu",
            "--batch-size",
            "1",
            "--sequence-length",
            "16",
            "--d-model",
            "32",
            "--n-heads",
            "4",
            "--n-kv-heads",
            "2",
            "--d-ff",
            "64",
            "--byte-layers",
            "1",
            "--byte-window",
            "8",
            "--patch-layers",
            "1",
            "--decoder-layers",
            "1",
            "--cpla-content-dim",
            "8",
            "--cpla-position-dim",
            "8",
            "--depths",
            "0",
            "1",
            "--default-depth",
            "1",
            "--log-every",
            "1",
            "--eval-every",
            "1",
            "--eval-batches",
            "1",
        ]
    )
    report = args.function(args)
    assert report["rounds_completed"] == 1
    assert (output / "thompson_state.json").exists()
    assert (output / "allocations.jsonl").exists()
    assert (output / "arms" / "byte-gqa" / "checkpoint.pt").exists()

    args.reward_cache_weight = 0.9
    with pytest.raises(ValueError, match="different shared settings or reward coefficients"):
        args.function(args)


def test_research_documents_are_clean_and_versioned() -> None:
    root = Path(__file__).resolve().parents[1]
    for relative in ("README.md", "docs/MATHEMATICAL_RESEARCH_NOTEBOOK.md"):
        text = (root / relative).read_text(encoding="utf-8")
        assert "CARS-R" in text
        assert not any(ord(character) < 32 and character not in "\t\n\r" for character in text)
    assert CARSRModel.checkpoint_version == "0.1.0"

"""
Unit tests for custom HuggingFace Trainer callbacks.
"""

from unittest.mock import MagicMock, patch

import pytest
import torch
from transformers import TrainerControl, TrainerState, TrainingArguments

from src.training.callbacks import EncoderUnfreezeCallback


def _make_state(global_step: int = 0) -> TrainerState:
    state = TrainerState()
    state.global_step = global_step
    return state


def _make_args() -> TrainingArguments:
    """Minimal TrainingArguments — output_dir required, everything else defaults."""
    return TrainingArguments(output_dir="/tmp/test_encoder_cb", use_cpu=True)


def _make_model(param_requires_grad: bool = True):
    """Tiny fake model whose encoder has a single parameter."""
    model = MagicMock()
    param = torch.nn.Parameter(torch.randn(4), requires_grad=param_requires_grad)
    model.model.encoder.parameters.return_value = iter([param])
    return model, param


class TestEncoderUnfreezeCallback:
    """Unit tests for EncoderUnfreezeCallback."""

    @pytest.mark.unit
    def test_encoder_frozen_on_train_begin(self):
        """on_train_begin should freeze all encoder parameters (requires_grad=False)."""
        param = torch.nn.Parameter(torch.randn(4), requires_grad=True)
        model = MagicMock()
        model.model.encoder.parameters.return_value = [param]

        callback = EncoderUnfreezeCallback(freeze_steps=1000)
        callback.on_train_begin(
            args=_make_args(),
            state=_make_state(0),
            control=TrainerControl(),
            model=model,
        )

        assert not param.requires_grad, "Encoder param should be frozen after on_train_begin"

    @pytest.mark.unit
    def test_encoder_not_unfrozen_before_target_step(self):
        """on_step_end before the target step must NOT unfreeze the encoder."""
        param = torch.nn.Parameter(torch.randn(4), requires_grad=False)
        model = MagicMock()
        model.model.encoder.parameters.return_value = [param]

        callback = EncoderUnfreezeCallback(freeze_steps=1000)
        callback.on_step_end(
            args=_make_args(),
            state=_make_state(global_step=999),
            control=TrainerControl(),
            model=model,
        )

        assert not param.requires_grad, "Encoder should remain frozen before target step"

    @pytest.mark.unit
    def test_encoder_unfrozen_at_target_step(self):
        """on_step_end at exactly freeze_steps should unfreeze the encoder."""
        param = torch.nn.Parameter(torch.randn(4), requires_grad=False)
        model = MagicMock()
        model.model.encoder.parameters.return_value = [param]

        callback = EncoderUnfreezeCallback(freeze_steps=1000)
        callback.on_step_end(
            args=_make_args(),
            state=_make_state(global_step=1000),
            control=TrainerControl(),
            model=model,
        )

        assert param.requires_grad, "Encoder should be unfrozen at the target step"

    @pytest.mark.unit
    def test_unfreeze_happens_only_once(self):
        """on_step_end calls after the unfreeze step must not call parameters() again."""
        param = torch.nn.Parameter(torch.randn(4), requires_grad=False)
        model = MagicMock()
        model.model.encoder.parameters.return_value = [param]

        callback = EncoderUnfreezeCallback(freeze_steps=500)
        # First call at target step — unfreezes
        callback.on_step_end(
            args=_make_args(),
            state=_make_state(global_step=500),
            control=TrainerControl(),
            model=model,
        )
        assert callback._unfrozen

        # Re-set the parameter to frozen to verify second call does not touch it
        param.requires_grad = False
        callback.on_step_end(
            args=_make_args(),
            state=_make_state(global_step=600),
            control=TrainerControl(),
            model=model,
        )
        # Guard prevents any action — param stays frozen
        assert not param.requires_grad, "Second call should be a no-op once _unfrozen=True"

    @pytest.mark.unit
    def test_zero_freeze_steps_disables_callback(self):
        """freeze_steps=0 should make the callback completely inert."""
        param = torch.nn.Parameter(torch.randn(4), requires_grad=True)
        model = MagicMock()
        model.model.encoder.parameters.return_value = [param]

        callback = EncoderUnfreezeCallback(freeze_steps=0)
        callback.on_train_begin(
            args=_make_args(),
            state=_make_state(0),
            control=TrainerControl(),
            model=model,
        )
        # Should not touch parameters
        assert param.requires_grad, "freeze_steps=0 should leave encoder untouched"

    @pytest.mark.unit
    def test_no_model_does_not_raise(self):
        """Callback must handle model=None gracefully (e.g. during dry runs)."""
        callback = EncoderUnfreezeCallback(freeze_steps=1000)
        # Should not raise
        callback.on_train_begin(args=_make_args(), state=_make_state(0), control=TrainerControl(), model=None)
        callback.on_step_end(args=_make_args(), state=_make_state(1000), control=TrainerControl(), model=None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

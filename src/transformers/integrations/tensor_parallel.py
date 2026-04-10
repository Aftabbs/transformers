# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from torch.distributed.tensor import DTensor, Partial, Replicate, Shard, distribute_tensor
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)
from torch.distributed.tensor.parallel.style import ParallelStyle
from torch.distributed.tensor.placement_types import _StridedShard

from ..utils import logging
from ..utils.import_utils import is_torch_available


if is_torch_available():
    import torch
    import torch.distributed as dist

    # Cache this result has it's a C FFI call which can be pretty time-consuming
    _torch_distributed_available = torch.distributed.is_available()


logger = logging.get_logger(__name__)


def replace_layer_number_by_wildcard(name: str) -> str:
    """
    Replace the numbers in the `name` by wildcards, only if they are in-between dots (`.`) or if they are between
    a dot (`.`) and the end of the string.
    This matches how modules are named/numbered when using a nn.ModuleList or nn.Sequential, but will NOT match
    numbers in a parameter name itself, e.g. if the param is named `"w1"` or `"w2"`.
    """
    return re.sub(r"\.\d+(\.|$)", lambda m: ".*" + m.group(1), name)


def _get_parameter_tp_plan(parameter_name: str, tp_plan: dict[str, str], is_weight=True) -> str | None:
    """
    Get the TP style for a parameter from the TP plan.

    The TP plan is a dictionary that maps parameter names to TP styles.
    The parameter name can be a generic name with wildcards (e.g. "*.weight") or a specific name (e.g. "layer_1.weight").

    The `is_weight` is important because for weights, we want to support `.weights` and `.bias` cases seamlessly! but
    not parent classes for `post_init` calls
    """
    generic_param_name = replace_layer_number_by_wildcard(parameter_name)
    if generic_param_name in tp_plan:
        return tp_plan[generic_param_name]
    elif is_weight and "." in generic_param_name and (module_name := generic_param_name.rsplit(".", 1)[0]) in tp_plan:
        return tp_plan[module_name]
    return None


# =============================================================================
# Tensor Sharding Utilities
# =============================================================================



# =============================================================================
# High-Level API Functions
# =============================================================================


def _to_cpu_fresh(tensor: torch.Tensor) -> torch.Tensor:
    """Plain tensor → contiguous CPU tensor with fresh storage for safetensors."""
    if tensor.device.type == "meta":
        return tensor
    t = tensor.detach()
    if t.device.type != "cpu":
        t = t.to(device="cpu")
    out = torch.empty(t.shape, dtype=t.dtype, device="cpu")
    out.copy_(t)
    return out.contiguous()


def gather_full_state_dict(model) -> dict[str, torch.Tensor]:
    """Gather all sharded params to full plain tensors for saving.

    Handles FSDP unshard and TP DTensor gather.
    Streams one parameter at a time to avoid holding all full tensors on GPU.
    Only rank 0 accumulates the result; other ranks return ``{}``.
    """
    tp_size = model.tp_size
    is_rank0 = dist.get_rank() == 0

    # Get state dict — FSDP unshard if needed (returns DTensors, not full tensors)
    if getattr(model, "_is_fsdp_managed_module", False):
        from torch.distributed.checkpoint.state_dict import get_model_state_dict

        state_dict = get_model_state_dict(model)
    else:
        state_dict = model.state_dict()

    # No TP — materialize on rank 0 only
    if tp_size is None:
        if is_rank0:
            return {k: _to_cpu_fresh(v) for k, v in state_dict.items()}
        return {}

    # Stream: gather one param at a time, only rank 0 keeps the CPU copy
    result = {}
    for key, tensor in state_dict.items():
        if isinstance(tensor, DTensor):
            # All ranks participate in the collective, only rank 0 keeps the result
            with torch.no_grad():
                full = tensor.redistribute(placements=[Replicate()] * tensor.device_mesh.ndim, async_op=False).to_local()
            if is_rank0:
                result[key] = _to_cpu_fresh(full)
            del full
        elif is_rank0:
            result[key] = _to_cpu_fresh(tensor)

    return result


def verify_tp_plan(expected_keys: list[str], tp_plan: dict[str, str | TPStyle] | None):
    """
    Verify the TP plan of the model, log a warning if the layers that were not sharded and the rules that were not applied.

    Only weight-sharding rules (colwise, rowwise, vocab, moe_experts) are checked.
    Module/activation entries (e.g. PrepareModuleInput, SequenceParallel) set up
    communication hooks on modules, not weight sharding, so they are excluded.
    """

    if tp_plan is None:
        return

    # Filter out module-level comm hooks — they don't shard weights
    _NON_WEIGHT_KINDS = {"activation", "module"}
    weight_plan = {
        k: v
        for k, v in tp_plan.items()
        if not isinstance(v, TPStyle) or v.kind not in _NON_WEIGHT_KINDS
    }

    generic_keys = {replace_layer_number_by_wildcard(key) for key in expected_keys}
    unsharded_layers = set(generic_keys)
    unused_rules = weight_plan.copy()

    for key in generic_keys:
        param_name = key.rsplit(".", 1)[0] if "." in key else key
        generic_param_name = re.sub(r"\d+", "*", param_name)

        if generic_param_name in weight_plan:
            unused_rules.pop(generic_param_name, None)
            unsharded_layers.discard(key)
        elif "." in generic_param_name and (parent_param_name := generic_param_name.rsplit(".", 1)[0]) in weight_plan:
            unused_rules.pop(parent_param_name, None)
            unsharded_layers.discard(key)

    if len(unused_rules) > 0:
        logger.warning(f"The following TP rules were not applied on any of the layers: {unused_rules}")
    if len(unsharded_layers) > 0:
        logger.warning(f"The following layers were not sharded: {', '.join(unsharded_layers)}")

class PrepareModuleInputOutput(ParallelStyle):
    """Allgather input (Shard(1) → Replicate) + local split output (Replicate → Shard(1)).

    Used for MoE blocks with SP: the input sequence is gathered before routing,
    and the output (after expert allreduce) is split back to match the residual.
    Forward output split is a local op (no comm). Backward creates the all-gather.
    """

    def __init__(self, use_local_output=True):
        super().__init__()
        self.use_local_output = use_local_output

    def _apply(self, module, device_mesh):
        def input_hook(mod, inputs):
            x = inputs[0] if isinstance(inputs, tuple) else inputs
            if not isinstance(x, DTensor):
                x = DTensor.from_local(x, device_mesh, [Shard(1)], run_check=False)
            x = x.redistribute(placements=[Replicate()])
            x = x.to_local()
            return (x,) + (inputs[1:] if isinstance(inputs, tuple) else ())

        def output_hook(mod, inputs, output):
            if not isinstance(output, DTensor):
                output = DTensor.from_local(output, device_mesh, [Replicate()], run_check=False)
            output = output.redistribute(placements=[Shard(1)])
            return output.to_local()

        module.register_forward_pre_hook(input_hook)
        module.register_forward_hook(output_hook)
        return module


# Maps string tp_plan entries for MoE experts to DTensor placements.
# Used by MoEExpertsParallel._partition_fn to create DTensors from the config plan.
_STRING_TO_PLACEMENT = {
    "packed_colwise": lambda: _StridedShard(dim=-2, split_factor=2),
    "colwise": lambda: Shard(-2),
    "rowwise": lambda: Shard(-1),
}



class MoEExpertsParallel(ParallelStyle):
    """Hybrid parallel style for MoE expert modules.

    Weights are converted to DTensors based on the per-parameter string plan
    entries (e.g. ``"packed_colwise"``, ``"rowwise"``) attached to the module
    by ``apply_tensor_parallel`` as ``_moe_param_plan``.
    Communication uses DTensor ``from_local``/``to_local`` on activations only —
    compatible with ``grouped_mm``.
    """

    def __init__(self, output_layouts=None):
        super().__init__()
        self.output_layouts = output_layouts or Replicate()

    @staticmethod
    def _partition_fn(name, module, device_mesh):
        param_plan = getattr(module, "_moe_param_plan", {})
        for param_name, param in module.named_parameters(recurse=False):
            plan_str = param_plan.get(param_name)
            if plan_str is None:
                continue
            placement_fn = _STRING_TO_PLACEMENT.get(plan_str)
            if placement_fn is None:
                continue
            placement = placement_fn()
            dtensor = distribute_tensor(param.data, device_mesh, [placement])
            module._parameters[param_name] = torch.nn.Parameter(dtensor, requires_grad=param.requires_grad)

    @staticmethod
    def _uses_partial_outputs(mod) -> bool:
        cached = getattr(mod, "_moe_outputs_are_partial", None)
        if cached is not None:
            return cached

        # Under TP-only the expert MLP dimension is sharded, so each rank emits a
        # partial hidden-state contribution that must be reduced. Under TP+FSDP,
        # FSDP can swap in full gathered expert weights for the current rank's
        # forward, in which case the local output is already complete.
        if hasattr(mod, "gate_up_proj"):
            gate_up_proj = mod.gate_up_proj.to_local() if isinstance(mod.gate_up_proj, DTensor) else mod.gate_up_proj
            full_expert_out = 2 * mod.intermediate_dim
            sharded_dim = -1 if getattr(mod, "is_transposed", False) else -2
            cached = gate_up_proj.shape[sharded_dim] != full_expert_out
        elif hasattr(mod, "up_proj"):
            up_proj = mod.up_proj.to_local() if isinstance(mod.up_proj, DTensor) else mod.up_proj
            full_expert_out = mod.intermediate_dim
            sharded_dim = -1 if getattr(mod, "is_transposed", False) else -2
            cached = up_proj.shape[sharded_dim] != full_expert_out
        else:
            cached = True

        mod._moe_outputs_are_partial = cached
        return cached

    @staticmethod
    def _prepare_input_fn(mod, inputs, device_mesh):
        hidden_states, top_k_index, top_k_weights = inputs[0], inputs[1], inputs[2]
        # from_local([Replicate()]).to_local(): forward sees plain tensor,
        # backward graph goes through DTensor all-reduce on gradient.
        if not isinstance(hidden_states, DTensor):
            hidden_states = DTensor.from_local(hidden_states, device_mesh, [Replicate()], run_check=False)
        hidden_states = hidden_states.to_local()
        if not isinstance(top_k_weights, DTensor):
            top_k_weights = DTensor.from_local(top_k_weights, device_mesh, [Replicate()], run_check=False)
        top_k_weights = top_k_weights.to_local()
        local_param_shadows = {}
        for param_name, param in list(mod.named_parameters(recurse=False)):
            if isinstance(param, DTensor):
                # grouped_mm expects plain tensors, but we must restore the
                # original DTensor params after the forward so save_pretrained
                # still sees the canonical sharded weights.
                local_param_shadows[param_name] = param
                mod._parameters.pop(param_name)
                setattr(mod, param_name, param.to_local())
        if local_param_shadows:
            shadow_stack = getattr(mod, "_moe_local_param_shadows", None)
            if shadow_stack is None:
                shadow_stack = []
                mod._moe_local_param_shadows = shadow_stack
            shadow_stack.append(local_param_shadows)
        return (hidden_states, top_k_index, top_k_weights)

    @staticmethod
    def _prepare_output_fn(output_layouts, mod, outputs, device_mesh):
        shadow_stack = getattr(mod, "_moe_local_param_shadows", None)
        if shadow_stack:
            for param_name, param in shadow_stack.pop().items():
                if hasattr(mod, param_name):
                    delattr(mod, param_name)
                mod.register_parameter(param_name, param)
        if outputs is None:
            return None
        # Plain TP expert weights produce partial outputs that need an all-reduce.
        # TP+FSDP can leave experts replicated across TP and sharded only across
        # experts/FSDP, in which case the local output is already complete.
        source_layout = Partial() if MoEExpertsParallel._uses_partial_outputs(mod) else Replicate()
        if not isinstance(outputs, DTensor):
            outputs = DTensor.from_local(outputs, device_mesh, [source_layout], run_check=False)
        # MoE experts output 2D [num_tokens, hidden]. For SP reduce-scatter,
        # Shard(1) means sequence dim in 3D, but in 2D the token dim is 0.
        actual_layouts = output_layouts
        if outputs.dim() == 2 and isinstance(output_layouts, Shard) and output_layouts.dim == 1:
            actual_layouts = Shard(0)
        if outputs.placements != (actual_layouts,):
            outputs = outputs.redistribute(placements=(actual_layouts,))
        return outputs.to_local()

    def _apply(self, module, device_mesh):
        # Don't use PyTorch's distribute_module — it would auto-convert all
        # params to Replicate DTensors. We create DTensors with proper Shard
        # placements in _partition_fn instead, and register hooks manually.
        self._partition_fn(module.__class__.__name__, module, device_mesh)
        module.register_forward_pre_hook(lambda mod, inputs: self._prepare_input_fn(mod, inputs, device_mesh))
        module.register_forward_hook(
            lambda mod, inputs, outputs: self._prepare_output_fn(self.output_layouts, mod, outputs, device_mesh),
            always_call=True,
        )
        return module


@dataclass(frozen=True)
class TPStyle:
    kind: Literal["colwise", "rowwise", "vocab", "activation", "module", "moe_experts"]
    comm: Literal["none", "allreduce", "reduce_scatter", "allgather", "allgather_split", "loss_parallel"]
    sequence_dim: int = 1
    use_local_output: bool = True
    input_key: str | None = None

    def to_dtensor_style(self) -> ParallelStyle:
        """Convert to the corresponding PyTorch DTensor ParallelStyle."""
        if self.kind == "colwise":
            match self.comm:
                case "none":
                    return ColwiseParallel(
                        input_layouts=Replicate(), output_layouts=Shard(-1), use_local_output=self.use_local_output
                    )
                case "allgather":
                    return ColwiseParallel(
                        input_layouts=Replicate(),
                        output_layouts=Replicate(),
                        use_local_output=self.use_local_output,
                    )
                case "loss_parallel":
                    return ColwiseParallel(input_layouts=Shard(1), output_layouts=Shard(-1), use_local_output=False)
        elif self.kind == "rowwise":
            match self.comm:
                case "allreduce":
                    return RowwiseParallel(
                        input_layouts=Shard(-1),
                        output_layouts=Replicate(),
                        use_local_output=self.use_local_output,
                    )
                case "reduce_scatter":
                    return RowwiseParallel(
                        input_layouts=Shard(-1), output_layouts=Shard(1), use_local_output=self.use_local_output
                    )
        elif self.kind == "vocab":
            match self.comm:
                case "allreduce":
                    return RowwiseParallel(
                        input_layouts=Replicate(),
                        output_layouts=Replicate(),
                        use_local_output=self.use_local_output,
                    )
                case "reduce_scatter":
                    return RowwiseParallel(
                        input_layouts=Replicate(), output_layouts=Shard(1), use_local_output=self.use_local_output
                    )
        elif self.kind == "activation":
            match self.comm:
                case "none":
                    return SequenceParallel(sequence_dim=self.sequence_dim, use_local_output=self.use_local_output)
        elif self.kind == "module":
            match self.comm:
                case "allgather":
                    if self.input_key is not None:
                        return PrepareModuleInput(
                            input_kwarg_layouts={self.input_key: Shard(1)},
                            desired_input_kwarg_layouts={self.input_key: Replicate()},
                            use_local_output=self.use_local_output,
                        )
                    return PrepareModuleInput(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),),
                        use_local_output=self.use_local_output,
                    )
                case "allgather_split":
                    return PrepareModuleInputOutput(use_local_output=self.use_local_output)
        elif self.kind == "moe_experts":
            match self.comm:
                case "allreduce":
                    return MoEExpertsParallel(output_layouts=Replicate())
                case "reduce_scatter":
                    return MoEExpertsParallel(output_layouts=Shard(1))
        raise ValueError(
            f"Invalid TPStyle({self.kind!r}, {self.comm!r}). Valid combinations:\n"
            f"  colwise:     none, allgather, loss_parallel\n"
            f"  rowwise:     allreduce, reduce_scatter\n"
            f"  vocab:       allreduce, reduce_scatter\n"
            f"  activation:  none\n"
            f"  module:      allgather, allgather_split\n"
            f"  moe_experts: allreduce, reduce_scatter"
        )

    def __str__(self):
        if self.comm == "none":
            return self.kind
        return f"{self.kind}_{self.comm}"


def apply_tensor_parallel(model, tp_mesh, tp_plan):
    """Apply tensor parallelism using PyTorch's parallelize_module.

    Converts the wildcard tp_plan from model config into a concrete plan
    for ``parallelize_module``. Plan values is a `TPStyle`` instances
    """
    if tp_plan is None:
        return model

    if tp_plan == "auto":
        enable_sp = getattr(getattr(model.config, "distributed_config", None), "enable_sequence_parallel", False)
        if enable_sp and hasattr(model.config, "base_model_sp_plan"):
            base_plan = model.config.base_model_sp_plan
        else:
            base_plan = model.config.base_model_tp_plan or {}

        # Prefix base model keys (e.g. "layers.*.q_proj" → "model.layers.*.q_proj")
        # Top-level keys like "lm_head" are kept as-is.
        base_model_prefix = model.base_model_prefix
        tp_plan = {}
        for k, v in base_plan.items():
            is_top_level = hasattr(model, k.split(".")[0])
            tp_plan[k if is_top_level else f"{base_model_prefix}.{k}"] = v

    parallelize_plan = {}

    for name, _ in model.named_modules():
        style_value = _get_parameter_tp_plan(parameter_name=name, tp_plan=tp_plan, is_weight=False)
        if style_value is None:
            continue

        if isinstance(style_value, TPStyle):
            parallelize_plan[name] = style_value.to_dtensor_style()
        elif isinstance(style_value, str):
            # String entries (e.g. "packed_colwise", "rowwise") are for parameter-level
            # shard-on-read during loading, not for parallelize_module. Skip them.
            continue
        else:
            parallelize_plan[name] = style_value

    # For MoE modules, collect per-parameter string plan entries and attach them
    # so _partition_fn can create DTensors with the correct placements.
    for name, mod in model.named_modules():
        if name in parallelize_plan and isinstance(parallelize_plan[name], MoEExpertsParallel):
            param_plan = {}
            for pname, _ in mod.named_parameters(recurse=False):
                child_style = _get_parameter_tp_plan(f"{name}.{pname}", tp_plan)
                if isinstance(child_style, str):
                    param_plan[pname] = child_style
            mod._moe_param_plan = param_plan

    parallelize_module(model, tp_mesh, parallelize_plan)

    return model

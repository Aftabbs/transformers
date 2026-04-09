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

import math
import re
from dataclasses import dataclass
from typing import Literal

from torch.distributed.tensor import DTensor, Partial, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)
from torch.distributed.tensor.parallel.style import ParallelStyle

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


if is_torch_available():
    str_to_dtype = {
        "BOOL": torch.bool,
        "U8": torch.uint8,
        "I8": torch.int8,
        "I16": torch.int16,
        "F16": torch.float16,
        "BF16": torch.bfloat16,
        "I32": torch.int32,
        "F32": torch.float32,
        "F64": torch.float64,
        "I64": torch.int64,
        "F8_E4M3": torch.float8_e4m3fn,
    }


def _blocks_to_block_sizes(total_size: int, blocks: int | list[int]) -> list[int]:
    """
    Convert block count or proportions to block sizes.

    This function accepts

    - The number of blocks (int), in which case the block size is
      total_size//blocks; or
    - A list of block sizes (list[int]).

    In the second case, if sum(blocks) < total_size, the ratios between
    the block sizes will be preserved. For instance, if blocks is
    [2, 1, 1] and total_size is 1024, the returned block sizes are
    [512, 256, 256].
    """
    if isinstance(blocks, list):
        total_blocks = sum(blocks)
        assert total_size % total_blocks == 0, f"Cannot split {total_size} in proportional blocks: {blocks}"
        part_size = total_size // total_blocks
        return [part_size * block for block in blocks]
    else:
        assert total_size % blocks == 0, f"Prepacked is not divisible by {blocks}"
        single_size = total_size // blocks
        return [single_size] * blocks


def get_packed_weights(param, empty_param, device_mesh, rank, dim):
    """
    When weights are packed (gate_up_proj), we need to make sure each shard gets its correct share.
    So if you have: gate_proj       ( 16, 5120, 8190)
    and             up_proj         ( 16, 5120, 8190)
    packed as       gate_up_proj    ( 16, 5120, 2 * 8190)
    And you shard along the last dimension, you need to interleave the gate and up values:

    Now, if we shard along the last dimension across TP_size (Tensor Parallelism size), we must interleave the values from gate and up projections correctly.

    Let's take TP_size = 4 for an example:

    Packed tensor `gate_up_proj`
    ---------------------------------------------------------------
    [ G0  G1  G2  G3 | G4  G5  G6  G7 | ... | U0  U1  U2  U3 | U4  U5  U6  U7 | ... ]
     ↑─────────────↑   ↑─────────────↑        ↑─────────────↑  ↑─────────────↑
       Gate Slice 0      Gate Slice 1            Up Slice 0       Up Slice 1

    Explanation:
    - The first half of the tensor (left of the center) holds the gate_proj values.
    - The second half (right of the center) holds the up_proj values.
    - For TP=4, we divide each half into 4 slices. In this example, we show two slices for brevity.
    - Each shard receives one slice from the gate part and the corresponding slice from the up part.

    For instance:
    • Shard 0 gets: [ Gate Slice 0, Up Slice 0 ] = [ G0, G1, G2, G3, U0, U1, U2, U3 ]
    • Shard 1 gets: [ Gate Slice 1, Up Slice 1 ] = [ G4, G5, G6, G7, U4, U5, U6, U7 ]
    • … and so on.

    This ensures that each shard receives an equal portion of both gate and up projections, maintaining consistency across tensor parallelism.
    """
    slice_ = param
    total_size = empty_param.shape[dim]
    world_size = device_mesh.size()
    block_sizes = _blocks_to_block_sizes(total_size=total_size, blocks=2)

    tensors_slices = []
    block_offset = 0
    for block_size in block_sizes:
        shard_block_size = block_size // world_size
        start = rank * shard_block_size
        stop = (rank + 1) * shard_block_size
        tensors_slices += range(block_offset + start, block_offset + stop)
        block_offset += block_size

    slice_dtype = slice_.get_dtype()
    # Handle F8_E4M3 dtype by converting to float16 before slicing
    # Without upcasting, the slicing causes : RuntimeError: "index_cpu" not implemented for 'Float8_e4m3fn'
    casted = False
    if slice_dtype == "F8_E4M3" or slice_dtype == "F8_E5M2":
        slice_ = slice_[...].to(torch.float16)
        casted = True

    if dim == 0:
        tensor = slice_[tensors_slices, ...]
    elif dim == 1 or dim == -2:
        tensor = slice_[:, tensors_slices, ...]
    elif dim == 2 or dim == -1:
        tensor = slice_[..., tensors_slices]
    else:
        raise ValueError(f"Unsupported dim {dim}, only dim 0, 1 or 2 are supported")

    if casted:
        return tensor
    else:
        return tensor.to(str_to_dtype[slice_dtype])


def repack_weights(
    packed_parameter: torch.Tensor,
    sharded_dim: int,  # The dimension index in the global tensor that was sharded
    world_size: int,
    num_blocks: int = 2,
) -> torch.Tensor:
    """
    Reorders a tensor that was reconstructed from sharded packed weights into its canonical packed format.

    For example, if a weight was packed (e.g., gate_proj and up_proj) and then sharded,
    DTensor.full_tensor() might produce an interleaved layout like [G0, U0, G1, U1, ...]
    along the sharded dimension. This function reorders it to [G0, G1, ..., U0, U1, ...].
    This is an inverse operation to get_packed_weights.

    Args:
        reconstructed_tensor: The tensor reconstructed from DTensor (e.g., via .full_tensor().contiguous()).
        sharded_dim: The dimension index in the reconstructed_tensor that was originally sharded.
        world_size: The tensor parallel world size.
        num_packed_projs: The number of projections that were packed together (e.g., 2 for gate_up_proj).

    Returns:
        The reordered tensor in canonical packed format.
    """

    if num_blocks != 2:
        raise ValueError(
            "Num blocks different from 2 is not supported yet. This is most likely a bug in your implementation as we only pack gate and up projections together."
        )

    actual_sharded_dim = sharded_dim if sharded_dim >= 0 else sharded_dim + packed_parameter.ndim
    total_size_on_sharded_dim = packed_parameter.shape[actual_sharded_dim]
    original_block_size_on_dim = total_size_on_sharded_dim // num_blocks
    shard_chunk_size = original_block_size_on_dim // world_size

    prefix_shape = packed_parameter.shape[:actual_sharded_dim]
    suffix_shape = packed_parameter.shape[actual_sharded_dim + 1 :]

    tensor_view = packed_parameter.view(
        *prefix_shape,
        world_size,
        num_blocks,
        shard_chunk_size,
        *suffix_shape,
    )

    # Permute to bring num_packed_projs first, then world_size, then shard_chunk_size
    # This groups all chunks of G together, then all chunks of U together.
    # Target order of these middle dimensions: (num_packed_projs, world_size, shard_chunk_size)
    # Current order of view's middle dimensions: (world_size, num_packed_projs, shard_chunk_size)
    # Absolute indices of the dimensions to be permuted (world_size, num_packed_projs)
    axis_ws_abs = len(prefix_shape)
    axis_npp_abs = len(prefix_shape) + 1

    permute_order = list(range(tensor_view.ndim))
    permute_order[axis_ws_abs], permute_order[axis_npp_abs] = permute_order[axis_npp_abs], permute_order[axis_ws_abs]

    tensor_permuted = tensor_view.permute(*permute_order)

    # Reshape back to the original tensor's ndim, with the sharded dimension now correctly ordered as [G_all, U_all].
    # The final shape should be the same as reconstructed_tensor.
    final_ordered_tensor = tensor_permuted.reshape_as(packed_parameter)

    return final_ordered_tensor


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


# Maps string tp_plan entries (MoE experts) to their weight shard dimension.
_STRING_PLAN_TO_SHARD_DIM = {
    "packed_colwise": -2,
    "rowwise": -1,
}


def _gather_full_param(tensor, shard_dim: int | None, tp_mesh, tp_size: int, is_packed: bool) -> torch.Tensor:
    """Gather a single sharded parameter (DTensor or plain MoE tensor) into a full tensor.

    - DTensor: ``redistribute([Replicate()])`` (collective across the DTensor's mesh).
    - Plain tensor with a ``shard_dim``: ``all_gather`` on the TP process group.

    If ``is_packed``, repacks interleaved gate/up shards back to canonical layout.
    """
    if isinstance(tensor, DTensor):
        with torch.no_grad():
            full = tensor.redistribute(placements=[Replicate()] * tensor.device_mesh.ndim, async_op=False).to_local()

        # Only repack if the DTensor was actually sharded along the packed axis
        if is_packed and shard_dim is not None:
            for p in tensor.placements:
                if not p.is_replicate() and hasattr(p, "dim") and p.dim == shard_dim:
                    full = repack_weights(full, shard_dim, tp_size, 2)
                    break
    else:
        # Plain MoE tensor — manual all_gather
        world_size = tp_mesh.size()
        process_group = tp_mesh.get_group("tp") if "tp" in (tp_mesh.mesh_dim_names or {}) else None
        norm_dim = shard_dim + tensor.ndim if shard_dim < 0 else shard_dim
        gathered = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor.contiguous(), group=process_group)
        full = torch.cat(gathered, dim=norm_dim)
        if is_packed:
            full = repack_weights(full, shard_dim, tp_size, 2)

    return full


def gather_full_state_dict(model) -> dict[str, torch.Tensor]:
    """Gather all sharded params to full plain tensors for saving.

    Handles FSDP unshard, TP DTensor gather, and MoE plain tensor gather.
    Streams one parameter at a time to avoid holding all full tensors on GPU.
    Only rank 0 accumulates the result; other ranks return ``{}``.
    """
    tp_plan = getattr(model, "_tp_plan", {}) or {}
    device_mesh = model.device_mesh
    base_prefix = model.base_model_prefix
    tp_size = model.tp_size
    tp_mesh = device_mesh["tp"] if device_mesh.ndim > 1 else device_mesh
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

    #NOTE(3outeille): This for MoE plain tensors
    def lookup_plan(key):
        plan = _get_parameter_tp_plan(key, tp_plan)
        if plan is None and base_prefix and key.startswith(base_prefix + "."):
            plan = _get_parameter_tp_plan(key[len(base_prefix) + 1 :], tp_plan)
        return plan

    # Stream: gather one param at a time, only rank 0 keeps the CPU copy
    result = {}
    for key, tensor in state_dict.items():
        current_plan = lookup_plan(key)

        # Resolve shard dim for string plan entries (MoE experts)
        shard_dim = None
        if isinstance(current_plan, str) and current_plan in _STRING_PLAN_TO_SHARD_DIM:
            shard_dim = _STRING_PLAN_TO_SHARD_DIM[current_plan]
            if shard_dim < 0:
                shard_dim += len(tensor.shape)

        is_sharded = isinstance(tensor, DTensor) or (tp_mesh is not None and shard_dim is not None)
        is_packed = isinstance(current_plan, str) and "packed" in current_plan

        if is_sharded:
            # All ranks participate in the collective, only rank 0 keeps the result
            full = _gather_full_param(tensor, shard_dim, tp_mesh, tp_size, is_packed)
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


class MoEShardOperation:
    """Shards MoE expert weights during loading.

    Same ``shard_tensor`` interface as ``DtensorShardOperation`` so it plugs
    into ``ParallelMaterializationContext`` / ``spawn_parallel_materialize``.

    - Non-packed weights (down_proj): sharded on read
    - Packed weights (gate_up_proj): returned full — ``set_param_for_module``
      shards after the ``WeightConverter`` merges + concatenates.
    """

    def __init__(self, device_mesh, param_name, empty_param):
        self.device_mesh = device_mesh
        self.rank = device_mesh.get_local_rank()
        self.param_name = param_name
        self.empty_param = empty_param

    def shard_tensor(self, param, tensor_idx=None, device=None, dtype=None):
        # Shard each individual expert tensor on read (before merge/concatenation).
        # Individual w1/w3 are 2D and should shard their output features on dim 0.
        # Some checkpoints already store fused 3D expert tensors (e.g. gate_up_proj),
        # where the expert dimension is leading and must stay replicated.
        param_shape = list(param.shape) if isinstance(param, torch.Tensor) else param.get_shape()

        if "gate" in self.param_name or "up" in self.param_name:
            if len(param_shape) > 2:
                return get_packed_weights(param, self.empty_param, self.device_mesh, self.rank, dim=-2).to(
                    device=device, dtype=dtype
                )
            dim = 0
        else:
            dim = len(param_shape) - 1  # rowwise: shard input features (last dim)

        world_size = self.device_mesh.size()
        shard_size = math.ceil(param_shape[dim] / world_size)
        start = self.rank * shard_size
        end = min(start + shard_size, param_shape[dim])

        slices = [slice(None)] * len(param_shape)
        slices[dim] = slice(start, end)
        return param[tuple(slices)].to(device=device, dtype=dtype)


class MoEExpertsParallel(ParallelStyle):
    """Hybrid parallel style for MoE expert modules.

    Weights are plain tensors, sharded during loading via ``MoEShardOperation``.
    Communication uses DTensor ``from_local``/``to_local`` on activations only —
    compatible with ``grouped_mm``.
    """

    def __init__(self, output_layouts=None):
        super().__init__()
        self.output_layouts = output_layouts or Replicate()

    @staticmethod
    def _partition_fn(name, module, device_mesh):
        # Mark module and params for MoE sharding during loading.
        # No DTensor distribution — weights stay as plain Parameters.
        module._moe_tp_mesh = device_mesh
        for param_name, param in module.named_parameters(recurse=False):
            param._moe_shard_info = {"mesh": device_mesh, "param_name": param_name}

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
        return (hidden_states, top_k_index, top_k_weights)

    @staticmethod
    def _prepare_output_fn(output_layouts, mod, outputs, device_mesh):
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
        # Don't use PyTorch's distribute_module — it auto-converts all params
        # to Replicate DTensors, which breaks shard-on-read for MoE weights.
        # Register hooks manually instead.
        self._partition_fn(module.__class__.__name__, module, device_mesh)
        module.register_forward_pre_hook(lambda mod, inputs: self._prepare_input_fn(mod, inputs, device_mesh))
        module.register_forward_hook(
            lambda mod, inputs, outputs: self._prepare_output_fn(self.output_layouts, mod, outputs, device_mesh)
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
                case "none":          return ColwiseParallel(input_layouts=Replicate(), output_layouts=Shard(-1), use_local_output=self.use_local_output)
                case "allgather":     return ColwiseParallel(input_layouts=Replicate(), output_layouts=Replicate(), use_local_output=self.use_local_output)
                case "loss_parallel": return ColwiseParallel(input_layouts=Shard(1), output_layouts=Shard(-1), use_local_output=False)
        elif self.kind == "rowwise":
            match self.comm:
                case "allreduce":      return RowwiseParallel(input_layouts=Shard(-1), output_layouts=Replicate(), use_local_output=self.use_local_output)
                case "reduce_scatter": return RowwiseParallel(input_layouts=Shard(-1), output_layouts=Shard(1), use_local_output=self.use_local_output)
        elif self.kind == "vocab":
            match self.comm:
                case "allreduce":      return RowwiseParallel(input_layouts=Replicate(), output_layouts=Replicate(), use_local_output=self.use_local_output)
                case "reduce_scatter": return RowwiseParallel(input_layouts=Replicate(), output_layouts=Shard(1), use_local_output=self.use_local_output)
        elif self.kind == "activation":
            match self.comm:
                case "none": return SequenceParallel(sequence_dim=self.sequence_dim, use_local_output=self.use_local_output)
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
                case "allreduce":      return MoEExpertsParallel(output_layouts=Replicate())
                case "reduce_scatter": return MoEExpertsParallel(output_layouts=Shard(1))
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

    parallelize_module(model, tp_mesh, parallelize_plan)

    return model

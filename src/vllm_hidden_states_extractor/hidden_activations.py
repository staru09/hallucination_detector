"""
Hidden Activations Connector for vLLM.

Captures hidden activations from a configurable layer during inference,
stores them in a GPU buffer and returns a buffer handle in the API response.
"""

import os
from dataclasses import dataclass, field
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch

from vllm.v1.attention.backend import AttentionMetadata
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger

from vllm_hidden_states_extractor.gpu_buffer import (
    get_global_buffer,
    init_global_buffer,
)

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)

# ─── Global state for hook <-> connector communication ───
# The hook writes here, the connector reads from here.
# No locks needed: single-threaded GPU execution in vLLM worker.
#
# Nested structure: req_id -> {layer_idx -> [handle, ...]}
_pending_hidden_states: Dict[str, Dict[int, List[str]]] = defaultdict(lambda: defaultdict(list))
_current_request_token_counts: Dict[str, int] = {}  # req_id -> num_tokens_in_batch
_layer_indices: List[int] = [20]               # configurable, multiple layers
_capture_enabled: bool = False
_registered_model: Optional[torch.nn.Module] = None  # model ref for deferred hook registration
_registered_hook_layers: List[int] = []               # which layers already have hooks


def _make_activation_hook(layer_idx: int):
    """
    Create a forward hook for a specific layer.

    The hook captures the layer output tensor, stores it in the
    GPU buffer manager, and records the handle for the current request.

    For each forward pass, the batch tensor is sliced per-request using
    _current_request_token_counts (populated from SchedulerOutput.
    num_scheduled_tokens). Each request gets its own tensor slice stored
    as a separate buffer handle.

    IMPORTANT: No locks, no CPU sync, no torch-unsupported ops.
    The tensor stays on GPU.

    NOTE: We call get_global_buffer() at runtime (not via closure)
    to ensure we always use the current buffer instance, even if
    init_global_buffer() was called after hook registration.
    """
    def hook(module, input, output):
        if not _capture_enabled:
            return
        if not _current_request_token_counts:
            return

        # Extract hidden states from output
        if isinstance(output, tuple):
            hidden_states = output[0]
        else:
            hidden_states = output

        # Get the current buffer (not closure-captured)
        buffer = get_global_buffer()

        # Slice per-request using token counts.
        # _current_request_token_counts is ordered (Python 3.7+ dict),
        # matching the order tokens appear in the batch tensor.
        offset = 0
        for req_id, num_tokens in _current_request_token_counts.items():
            # Slice this request's tokens from the batch
            req_hidden = hidden_states[offset:offset + num_tokens]
            offset += num_tokens

            handle = buffer.store(
                req_hidden,
                metadata={
                    "req_id": req_id,
                    "layer_idx": layer_idx,
                    "shape": list(req_hidden.shape),
                    "dtype": str(req_hidden.dtype),
                    "is_prefill": num_tokens > 1,
                },
            )
            _pending_hidden_states[req_id][layer_idx].append(handle)

    return hook


def register_activation_hooks(model: torch.nn.Module, layer_indices: list[int]):
    """
    Register forward hooks on the specified layers of the model.

    Tracks which layers already have hooks to avoid duplicates.
    Stores the model reference so the connector can re-register
    hooks on additional layers later.

    Args:
        model: The model to register hooks on
        layer_indices: Which layers to extract activations from (e.g., [4, 8, 16, 20, 28])

    Returns:
        List of hook handles
    """
    global _registered_model, _registered_hook_layers
    handles = []

    # Find the layers module
    layers = None
    for name, module in model.named_modules():
        if name == 'model.layers' or name.endswith('.model.layers'):
            layers = module
            logger.info(f"Found layers at: {name}")
            break

    if layers is None:
        for name, module in model.named_modules():
            if 'layers' in name and isinstance(module, torch.nn.ModuleList):
                layers = module
                logger.info(f"Found layers at: {name}")
                break

    if layers is None:
        logger.error("Could not find model layers for hook registration")
        return handles

    # Store model reference for deferred registration
    _registered_model = model

    for layer_idx in layer_indices:
        if layer_idx in _registered_hook_layers:
            logger.info(f"Layer {layer_idx} already has a hook, skipping")
            continue
        if layer_idx < len(layers):
            layer = layers[layer_idx]
            hook_handle = layer.register_forward_hook(_make_activation_hook(layer_idx))
            handles.append(hook_handle)
            _registered_hook_layers.append(layer_idx)
            logger.info(f"Hidden activations hook registered on layer {layer_idx}")
        else:
            logger.error(f"Layer {layer_idx} out of range (model has {len(layers)} layers)")

    return handles


def ensure_hooks_registered():
    """
    Ensure hooks are registered on all layers in _layer_indices.

    Called by the connector after it sets _layer_indices, in case the
    model was loaded before the connector and only got partial hooks.
    """
    if _registered_model is None:
        logger.warning("No model registered yet, cannot ensure hooks")
        return []

    missing = [l for l in _layer_indices if l not in _registered_hook_layers]
    if not missing:
        logger.info(f"All hooks already registered on layers {_layer_indices}")
        return []

    logger.info(f"Registering missing hooks on layers {missing}")
    return register_activation_hooks(_registered_model, missing)


# ─── Connector ───

@dataclass
class ActivationsConnectorMetadata(KVConnectorMetadata):
    requests: list = field(default_factory=list)


class HiddenActivationsConnector(KVConnectorBase_V1):
    """
    KV Connector that captures hidden activations from model layers.

    Hidden states are stored in a GPU buffer (no CPU transfer).
    The buffer handle is returned in the API response via kv_transfer_params.

    Config (via kv_connector_extra_config):
        - activation_layers: list[int] = [20]  (which layers to capture)
        - activation_layer: int = 20           (single layer, backward compat)
        - buffer_size: int = 64                (max number of stored tensors)
        - buffer_ttl: float = 30.0             (seconds before auto-cleanup)
    """

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        global _layer_indices, _capture_enabled

        self._block_size = vllm_config.cache_config.block_size

        # Read config — support both plural and singular forms
        activation_layers = self._kv_transfer_config.get_from_extra_config(
            "activation_layers", None
        )
        if activation_layers is not None:
            _layer_indices = list(activation_layers)
        else:
            # Backward compat: single layer
            single_layer = self._kv_transfer_config.get_from_extra_config(
                "activation_layer", 20
            )
            _layer_indices = [single_layer]

        buffer_size = self._kv_transfer_config.get_from_extra_config(
            "buffer_size", 64
        )
        buffer_ttl = self._kv_transfer_config.get_from_extra_config(
            "buffer_ttl", 30.0
        )

        # Scale buffer by number of layers
        scaled_buffer_size = buffer_size * len(_layer_indices)

        # Initialize global buffer
        init_global_buffer(
            max_slots=scaled_buffer_size,
            default_ttl=buffer_ttl,
        )

        # Enable hooks via env var for model patching
        os.environ["HIDDEN_ACTIVATIONS_ENABLED"] = "1"
        import json
        os.environ["HIDDEN_ACTIVATIONS_LAYERS"] = json.dumps(_layer_indices)
        # Backward compat env var
        os.environ["HIDDEN_ACTIVATIONS_LAYER"] = str(_layer_indices[0])

        _capture_enabled = True

        # If the model was loaded before us (common case), it may only
        # have hooks on the default layer. Register any missing layers.
        ensure_hooks_registered()

        # Auto-start real-time consumer if requested
        from vllm_hidden_states_extractor.realtime_consumer import maybe_auto_start
        self._consumer_stop_fn = maybe_auto_start()

        logger.info(
            f"HiddenActivationsConnector initialized: "
            f"layers={_layer_indices}, buffer_size={scaled_buffer_size}, ttl={buffer_ttl}s"
        )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register KV caches."""
        logger.info(f"Registered {len(kv_caches)} KV cache layers")

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        pass

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        pass

    def wait_for_save(self):
        return

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        return 0, False

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        pass

    def build_connector_meta(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> KVConnectorMetadata:
        """Track ALL active requests so the hook captures every step.

        Uses num_scheduled_tokens which includes both new prefill requests
        and ongoing decode requests, with their token counts per step.
        """
        global _current_request_token_counts
        meta = ActivationsConnectorMetadata()

        # num_scheduled_tokens: dict[str, int] — covers all active requests
        _current_request_token_counts = dict(scheduler_output.num_scheduled_tokens)

        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Return all buffer handles AND per-layer, per-step tensor info.

        Returns handles, shapes, and stats for every forward step
        (prefill + each decode step) per captured layer directly in
        kv_transfer_params, so no separate API is needed to inspect the tensors.
        """
        global _current_request_token_counts

        req_id = request.request_id
        layer_handles = dict(_pending_hidden_states.pop(req_id, {}))

        # Remove from active tracking
        _current_request_token_counts.pop(req_id, None)

        if layer_handles:
            buffer = get_global_buffer()

            # Build per-layer info
            per_layer = {}
            all_handles_flat = []
            total_tensors = 0

            for layer_idx in sorted(layer_handles.keys()):
                handles = layer_handles[layer_idx]
                all_handles_flat.extend(handles)
                steps = []
                layer_tensors = []

                for i, handle in enumerate(handles):
                    tensor, meta = buffer.get(handle)
                    if tensor is not None:
                        t_float = tensor.float()
                        steps.append({
                            "step": i,
                            "handle": handle,
                            "shape": list(tensor.shape),
                            "is_prefill": meta.get("is_prefill", False),
                            "min": round(t_float.min().item(), 6),
                            "max": round(t_float.max().item(), 6),
                            "mean": round(t_float.mean().item(), 6),
                            "std": round(t_float.std().item(), 6),
                        })
                        layer_tensors.append(tensor)

                if layer_tensors:
                    stacked = torch.cat(layer_tensors, dim=0)
                    per_layer[str(layer_idx)] = {
                        "handles": handles,
                        "num_steps": len(handles),
                        "steps": steps,
                        "stacked_shape": list(stacked.shape),
                        "total_tokens": stacked.shape[0],
                    }
                    total_tensors += len(layer_tensors)

            # Grab dtype/hidden_dim from first available tensor
            first_layer = next(iter(per_layer.values()), {})
            first_step = (first_layer.get("steps") or [{}])[0] if first_layer else {}
            dtype_str = ""
            hidden_dim = 0
            if per_layer:
                sample_layer_idx = int(next(iter(per_layer)))
                sample_handle = layer_handles[sample_layer_idx][0]
                sample_tensor, _ = buffer.get(sample_handle)
                if sample_tensor is not None:
                    dtype_str = str(sample_tensor.dtype)
                    hidden_dim = sample_tensor.shape[-1]

            logger.info(
                f"Request {req_id}: {total_tensors} hidden state tensors captured "
                f"across {len(per_layer)} layers ({sorted(int(k) for k in per_layer)})"
            )

            return False, {
                "hidden_states_handles": all_handles_flat,
                "hidden_states_layers": sorted(int(k) for k in per_layer.keys()),
                "hidden_states_num_layers": len(per_layer),
                "hidden_states_per_layer": per_layer,
                "hidden_states_dtype": dtype_str,
                "hidden_states_device": "gpu",
                "hidden_states_hidden_dim": hidden_dim,
            }
        else:
            logger.warning(f"No hidden states captured for request {req_id}")
            return False, {"hidden_states_handles": [], "hidden_states_layers": []}

    def clear_connector_metadata(self):
        pass

    def real_clear_connector_metadata(self):
        self._connector_metadata = None

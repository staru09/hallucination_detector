# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.v1.attention.backend import AttentionBackend, AttentionType
from vllm.attention.layer import get_attention_context

from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.attention.utils.kv_transfer_utils import maybe_transfer_kv_layer
from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.attention.layer import set_default_quant_scales
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.distributed.kv_transfer import (
    get_kv_transfer_group,
    has_kv_transfer_group,
    is_v1_kv_transfer_group,
)
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.multimodal.inputs import NestedTensors

from vllm.model_executor.models.utils import maybe_prefix
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec

from vllm_hidden_states_extractor.attention import CacheOnlyAttentionBackend
from vllm_hidden_states_extractor.utils import reshape_hidden_states_for_kv_cache

logger = init_logger(__name__)


# @support_torch_compile(
#     dynamic_arg_dims={
#         "input_ids": 0,
#         "positions": -1,
#         "hidden_states": 0,
#         "input_embeds": 0,
#     }
# )


class DummyModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.num_hidden_states = len(
            getattr(self.config, "eagle_aux_hidden_state_layer_ids", [])
        )

        self.embed_tokens = VocabParallelEmbedding(
            self.config.vocab_size,
            self.config.hidden_size // self.num_hidden_states,
            prefix=maybe_prefix(prefix, "embed_tokens"),
        )


class CacheOnlyAttentionLayer(nn.Module, AttentionLayerBase):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        logits_soft_cap: float | None = None,
        per_layer_sliding_window: int | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        attn_backend: type[AttentionBackend] | None = None,
        head_size_v: int | None = None,
        num_hidden_states: int = 1,
        **extra_impl_args,
    ):
        super().__init__()
        assert alibi_slopes is None, (
            "CacheOnlyAttention does not support alibi slopes yet."
        )
        assert logits_soft_cap is None, (
            "CacheOnlyAttention does not support logits soft cap yet."
        )
        assert per_layer_sliding_window is None, (
            "CacheOnlyAttention does not support per-layer sliding window yet."
        )
        assert attn_backend is None, (
            "CacheOnlyAttention does not support attn_backend yet."
        )
        assert head_size_v is None, (
            "CacheOnlyAttention does not support head_size_v yet."
        )
        assert kv_sharing_target_layer_name is None, (
            "CacheOnlyAttention does not support kv sharing yet."
        )

        vllm_config = get_current_vllm_config()

        cache_config = cache_config or vllm_config.cache_config
        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
            self.block_size = cache_config.block_size
            calculate_kv_scales = cache_config.calculate_kv_scales
        else:
            kv_cache_dtype = "auto"
            self.block_size = 16
            calculate_kv_scales = False
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            cache_config.cache_dtype, vllm_config.model_config
        )
        if num_kv_heads is None:
            num_kv_heads = num_heads
        assert num_heads % num_kv_heads == 0, (
            f"num_heads ({num_heads}) is not divisible by num_kv_heads ({num_kv_heads})"
        )
        self.quant_config = quant_config
        self.layer_name = prefix

        # Initialize KV cache quantization attributes
        set_default_quant_scales(self, register_buffer=True)
        # _init_kv_cache_quant(
        #     self,
        #     self.quant_config,
        #     self.layer_name,
        #     kv_cache_dtype,
        #     calculate_kv_scales,
        # )

        self.num_heads = num_heads
        self.head_size = head_size
        self.head_size_v = self.head_size if head_size_v is None else head_size_v
        self.num_kv_heads = num_kv_heads

        # NOTE: model_config may be None during certain tests
        model_config = vllm_config.model_config
        self.use_mm_prefix = model_config is not None and model_config.is_mm_prefix_lm

        # During model initialization, the default dtype is set as the model
        # weight and activation dtype.
        dtype = torch.get_default_dtype()

        self.attn_backend = CacheOnlyAttentionBackend

        impl_cls = self.attn_backend.get_impl_cls()
        self.impl = impl_cls(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            None,  # sliding_window
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            **extra_impl_args,
        )
        self.backend = AttentionBackendEnum[self.attn_backend.get_name()]
        self.dtype = dtype

        # use a placeholder kv cache tensor during init, which will be replaced
        # by bind_kv_cache
        # this variable will not be accessed if use_direct_call is True
        self.kv_cache = [
            torch.tensor([])
            for _ in range(vllm_config.parallel_config.pipeline_parallel_size)
        ]

        # tp_size = get_tensor_model_parallel_world_size()
        # if self.total_num_kv_heads >= tp_size:
        #     # Number of KV heads is greater than TP size, so we partition
        #     # the KV heads across multiple tensor parallel GPUs.
        #     assert self.total_num_kv_heads % tp_size == 0
        # else:
        #     # Number of KV heads is less than TP size, so we replicate
        #     # the KV heads across multiple tensor parallel GPUs.
        #     assert tp_size % self.total_num_kv_heads == 0
        # self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def forward(
        self,
        hidden_states: torch.Tensor,  # shape: [batch_size, hidden_size * num_hidden_states]
        output_shape: torch.Size | None = None,
    ):
        output_dtype = hidden_states.dtype
        if output_shape is None:
            # Handle both 2D [num_tokens, hidden] and
            # 3D [num_tokens, heads, head_dim] query
            num_tokens = hidden_states.shape[0]
            output_shape = torch.Size((num_tokens, self.num_heads * self.head_size_v))
        output_shape = output_shape if output_shape is not None else hidden_states.shape
        output = torch.empty(
            output_shape, dtype=output_dtype, device=hidden_states.device
        )
        hidden_size = output_shape[-1]
        output = output.view(-1, self.num_heads, self.head_size_v)

        key, value = reshape_hidden_states_for_kv_cache(hidden_states, self.head_size)

        # forward_context: ForwardContext = get_forward_context()
        # attn_metadata = forward_context.attn_metadata
        # if isinstance(attn_metadata, dict):
        #     attn_metadata = attn_metadata[self.layer_name]
        # self_kv_cache = self.kv_cache[forward_context.virtual_engine]

        # return self.impl.forward(
        #     self,
        #     None,
        #     key,
        #     value,
        #     self_kv_cache,
        #     attn_metadata,
        #     output=output
        # )
        cache_only_attention_with_kv_transfer(None, key, value, output, self.layer_name)
        return output.view(-1, hidden_size)

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return FullAttentionSpec(
            block_size=self.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            dtype=self.kv_cache_torch_dtype,
        )


@maybe_transfer_kv_layer
def cache_only_attention_with_kv_transfer(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
) -> None:
    attn_metadata, self, kv_cache = get_attention_context(layer_name)

    self.impl.forward(
        self,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=output,
        output_scale=output_scale,
        output_block_scale=output_block_scale,
    )


class HiddenStatesExtractor(Eagle3LlamaForCausalLM):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)
        print("HiddenStatesExtractor __init__")
        self.config = vllm_config.speculative_config.draft_model_config.hf_config
        self.num_hidden_states = len(
            getattr(self.config, "eagle_aux_hidden_state_layer_ids", [])
        )
        # Ensure draft_vocab_size is set
        # default to the base vocab size when absent
        if getattr(self.config, "draft_vocab_size", None) is None:
            base_vocab_size = getattr(self.config, "vocab_size", None)
            self.config.draft_vocab_size = base_vocab_size
        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )

        num_heads = self.config.num_attention_heads
        head_size = self.config.head_dim
        scale = head_size**-0.5
        num_kv_heads = self.config.num_key_value_heads
        cache_config = vllm_config.cache_config

        # Store target layer count in draft config for
        # proper layer_types indexing in draft models
        self.config.target_layer_count = target_layer_num

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size // self.num_hidden_states,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size, scale=logit_scale
        )
        self.draft_id_to_target_id = nn.Parameter(
            torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
            requires_grad=False,
        )

        self.layers = nn.ModuleList(
            [
                CacheOnlyAttentionLayer(
                    num_heads=2 * num_heads,
                    head_size=head_size,
                    scale=scale,
                    num_kv_heads=2 * num_kv_heads,
                    cache_config=cache_config,
                    prefix=maybe_prefix(
                        prefix, f"layers.{layer_idx + target_layer_num}"
                    ),
                    # attn_backend=CacheOnlyAttentionBackend,
                    num_hidden_states=self.num_hidden_states,
                )
                for layer_idx in range(self.config.num_hidden_layers)
            ]
        )
        self.model = DummyModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: NestedTensors | None = None,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # todo: Cache hidden states
        # hidden_states is (batch_size, hidden_size * num_hidden_states)

        # forward_context: ForwardContext = get_forward_context()
        # attn_metadata = forward_context.attn_metadata
        # self_kv_cache = self.layers[0].kv_cache[forward_context.virtual_engine]

        self.layers[0].forward(hidden_states)
        if has_kv_transfer_group() and is_v1_kv_transfer_group():
            kv_connector = get_kv_transfer_group()
            kv_connector.real_clear_connector_metadata()

        dummy_ret = hidden_states.new_zeros(
            (hidden_states.shape[0], hidden_states.shape[1] // self.num_hidden_states)
        )
        return dummy_ret, dummy_ret

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        if self.draft_id_to_target_id is None:
            assert logits.shape[1] == self.config.vocab_size, (
                "Expected logits to have shape "
                f"(*, {self.config.vocab_size}), but got {logits.shape}"
            )
            return logits

        base = torch.arange(self.config.draft_vocab_size, device=logits.device)
        targets = base + self.draft_id_to_target_id
        logits_new = logits.new_full(
            (
                logits.shape[0],
                self.config.vocab_size,
            ),
            float("-inf"),
        )
        logits_new[:, targets] = logits
        return logits_new

    def combine_hidden_states(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        # no-op. We return the full hidden states so that they are all passed into the forward fn where they will be cached.
        # Note: this requires setting the dummy model hidden size to (num_hidden_states * hidden_size)
        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        return None

from typing import ClassVar
import torch
from triton import cdiv
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionType,
    is_quantized_kv_cache,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.v1.attention.backends.utils import (
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from dataclasses import dataclass
from vllm.v1.kv_cache_interface import AttentionSpec


@register_backend(AttentionBackendEnum.CUSTOM)
class CacheOnlyAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = False
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
        # todo: expand to all dtypes? Or no dtypes?
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto"]

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        """CacheOnlyAttention supports decoder attention."""
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_mm_prefix(cls) -> bool:
        """CacheOnlyAttention does supports full attention for image tokens."""
        # todo: review this
        return True

    @staticmethod
    def get_impl_cls() -> type["CacheOnlyAttentionImpl"]:
        return CacheOnlyAttentionImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # todo: Check if we can just set this to (num_hidden_states, num_blocks, block_size, num_kv_heads, head_size)
        # or if we need to keep it with 2 and use multiple layers
        # What happens if kv cache size doesn't match other layers
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_builder_cls() -> type["CacheOnlyAttentionMetadataBuilder"]:
        return CacheOnlyAttentionMetadataBuilder

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # todo: review this? Does [] mean all head sizes?
        return []


@dataclass
class CacheOnlyAttentionMetadata:
    causal: bool
    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: torch.Tensor | None
    prefix_kv_lens: torch.Tensor | None
    suffix_kv_lens: torch.Tensor | None

    num_reqs: int

    # For logging.
    num_input_tokens: int = 0  # Number of tokens including padding.

    # todo: add CacheOnly Metadata if needed

    def __post_init__(self):
        # todo: copied from FlexAttentionMetadata, review if needed
        assert self.use_cascade is False, "Not implemented yet."
        assert self.common_prefix_len == 0, "Not implemented yet."
        assert self.cu_prefix_query_lens is None, "Not implemented yet."
        assert self.prefix_kv_lens is None, "Not implemented yet."
        assert self.suffix_kv_lens is None, "Not implemented yet."


class CacheOnlyAttentionMetadataBuilder(
    AttentionMetadataBuilder[CacheOnlyAttentionMetadata]
):
    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config

        self.num_heads_q = self.model_config.get_num_attention_heads(
            self.parallel_config
        )
        self.num_heads_kv = self.model_config.get_num_kv_heads(self.parallel_config)
        self.headdim = self.model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size
        self.kv_cache_spec = kv_cache_spec

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> CacheOnlyAttentionMetadata:
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len

        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        num_blocks_per_seq = cdiv(seq_lens, self.block_size)

        use_cascade = common_prefix_len > 0
        cu_prefix_query_lens = None
        prefix_kv_lens = None
        suffix_kv_lens = None
        if use_cascade:
            raise NotImplementedError("Not yet my friend")

        out = CacheOnlyAttentionMetadata(
            causal=common_attn_metadata.causal,
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            num_reqs=num_reqs,
        )
        return out

    def use_cascade_attention(self, *args, **kwargs) -> bool:
        return False


class CacheOnlyAttentionImpl(AttentionImpl):
    sliding_window: int | None
    alibi_slopes: torch.Tensor | None
    logits_soft_cap: float | None
    mm_prefix_range: dict[int, list[tuple[int, int]]] | None = None

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        **kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.attn_type = attn_type

        if attn_type not in (AttentionType.DECODER):
            raise NotImplementedError(
                f"CacheOnlyAttention does not support {attn_type} attention"
            )

        if alibi_slopes is not None:
            raise NotImplementedError(
                "CacheOnlyAttention does not support alibi slopes yet."
            )
        else:
            self.alibi_slopes = None

        self.sliding_window = sliding_window

        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = logits_soft_cap
        if self.logits_soft_cap is not None:
            raise NotImplementedError(
                "CacheOnlyAttention does not support logits soft cap yet."
            )

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError(
                "CacheOnlyAttention does not support kv sharing yet."
            )

        if is_quantized_kv_cache(self.kv_cache_dtype):
            # todo: maybe we can add support for this?
            raise NotImplementedError(
                "CacheOnlyAttention does not support quantized kv-cache. Yet"
            )

    # @staticmethod
    # def view_as_4d(tensor: torch.Tensor) -> torch.Tensor:
    #     """View a 3d tensor as 4D."""
    #     if tensor.ndim == 4:
    #         return tensor
    #     assert tensor.ndim == 3
    #     return tensor[None, :, :, :]

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: CacheOnlyAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with FLexAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for CacheOnlyAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)
            # query = self.view_as_4d(query).permute(0, 2, 1, 3)
            # return torch.empty_like(query)

        num_actual_tokens = attn_metadata.num_actual_tokens

        assert attn_metadata.causal, (
            "CacheOnlyAttention does not support non-causal attention yet."
        )
        assert self.attn_type == AttentionType.DECODER
        key_cache, value_cache = kv_cache.unbind(0)

        torch.ops._C_cache_ops.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            attn_metadata.slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

        # # View out the block_size dim
        # key_cache = key_cache.view(-1, self.num_kv_heads, self.head_size)
        # value_cache = value_cache.view(-1, self.num_kv_heads, self.head_size)
        # query, key_tensor, value_tensor = map(
        #     lambda x: self.view_as_4d(x).permute(0, 2, 1, 3),
        #     (query, key_cache, value_cache),
        # )

        # query = query[:, :, :num_actual_tokens, :]

        # # Doesn't work for now -> constraint violation
        # # torch._dynamo.try_mark_dynamic(query, 2)

        # assert attn_metadata.block_mask is not None
        # block_m, block_n = attn_metadata.block_mask.BLOCK_SIZE

        # kernel_options = get_kernel_options(
        #     query, block_m, block_n, attn_metadata.direct_build
        # )
        # out = flex_attention_compiled(
        #     query,
        #     key_tensor,
        #     value_tensor,
        #     attn_metadata.transformed_score_mod,
        #     attn_metadata.block_mask,
        #     self.scale,
        #     enable_gqa=enable_gqa,
        #     kernel_options=kernel_options,
        # )

        # # Flex doesn't have an out variant today, rely on epilogue fusion
        # out = out.permute(0, 2, 1, 3).squeeze(0)
        # output[:num_actual_tokens, :, :].copy_(out)

        return output.fill_(0)

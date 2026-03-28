import os
import functools


def register():
    from vllm import ModelRegistry
    from vllm.transformers_utils.configs.speculators.algos import (
        register_speculator,
        update_eagle3,
    )
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

    # ── Original speculative decoding approach ──
    @register_speculator("extract_hidden_states")
    def update_extract_hidden_states(config_dict: dict, vllm_config: dict) -> None:
        update_eagle3(config_dict, vllm_config)
        vllm_config["method"] = "eagle3"
        vllm_config["architectures"] = ["HiddenStatesExtractor"]

    print("HiddenStatesExtractor registered")
    if "HiddenStatesExtractor" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "HiddenStatesExtractor",
            "vllm_hidden_states_extractor.model:HiddenStatesExtractor",
        )

    if "ExampleHiddenStatesConnector" not in KVConnectorFactory._registry:
        KVConnectorFactory.register_connector(
            "ExampleHiddenStatesConnector",
            "vllm_hidden_states_extractor.connector",
            "ExampleHiddenStatesConnector",
        )

    # ── New GPU-resident activation connector ──
    if "HiddenActivationsConnector" not in KVConnectorFactory._registry:
        KVConnectorFactory.register_connector(
            "HiddenActivationsConnector",
            "vllm_hidden_states_extractor.hidden_activations",
            "HiddenActivationsConnector",
        )
        print("HiddenActivationsConnector registered")

    # Always patch model classes. The patch reads the layer list from the
    # hidden_activations module's _layer_indices at model init time (not at
    # patch time), so it works regardless of whether the connector has
    # initialized yet.
    _patch_model_for_activations()


def _patch_model_for_activations():
    """
    Monkey-patch supported model classes to register forward hooks
    on the target layers after initialization.

    The patch reads _layer_indices from hidden_activations at model init
    time (not at patch time), so it picks up whatever layers the connector
    has configured.

    Supports: Llama, Qwen3 (add more as needed)
    """
    models_to_patch = [
        ("vllm.model_executor.models.llama", "LlamaForCausalLM"),
        ("vllm.model_executor.models.qwen3", "Qwen3ForCausalLM"),
    ]

    for module_path, class_name in models_to_patch:
        try:
            import importlib
            mod = importlib.import_module(module_path)
            model_cls = getattr(mod, class_name)
            _apply_activation_patch(model_cls, class_name)
        except (ImportError, AttributeError):
            # Model not available in this vLLM install, skip
            pass


def _apply_activation_patch(model_cls, class_name: str):
    """Apply the activation hook patch to a single model class."""
    original_init = model_cls.__init__

    @functools.wraps(original_init)
    def patched_init(self, *, vllm_config, prefix: str = "", **kwargs):
        original_init(self, vllm_config=vllm_config, prefix=prefix, **kwargs)

        # Read layer list at model init time — the connector may have
        # updated _layer_indices by now
        from vllm_hidden_states_extractor.hidden_activations import (
            register_activation_hooks,
            _layer_indices,
        )

        # Also check env var in case connector set it
        import json
        layers_env = os.environ.get("HIDDEN_ACTIVATIONS_LAYERS", "")
        if layers_env:
            activation_layers = json.loads(layers_env)
        else:
            activation_layers = list(_layer_indices)

        try:
            handles = register_activation_hooks(self, activation_layers)
            self._activation_hook_handles = handles
            print(f"[HiddenActivations] Registered hooks on layers {activation_layers} of {class_name}")
        except Exception as e:
            print(f"[HiddenActivations] Warning: Failed to register hooks on {class_name}: {e}")

    model_cls.__init__ = patched_init
    print(f"[HiddenActivations] {class_name} patched for activation hooks")

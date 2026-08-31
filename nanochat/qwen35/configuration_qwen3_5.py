# Configuration for the text-only Qwen3.5 LLM used for nanochat-style training.
# Self-contained so it does not depend on `transformers` shipping a `qwen3_5` model.
# Field names for the linear (GatedDeltaNet) path mirror Qwen3-Next for compatibility.

from transformers.configuration_utils import PretrainedConfig


class Qwen3_5TextConfig(PretrainedConfig):
    """Configuration for the Qwen3.5 *text* model (LLM only, no vision).

    The architecture is a hybrid of GatedDeltaNet "linear_attention" layers and
    gated softmax "full_attention" layers, controlled by `layer_types` (or auto
    generated from `full_attention_interval`).
    """

    model_type = "qwen3_5_text"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=28,
        num_attention_heads=8,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=False,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        rope_parameters=None,
        partial_rotary_factor=1.0,
        mrope_section=None,
        attention_bias=False,
        attention_dropout=0.0,
        # GatedDeltaNet (linear attention) dims
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_value_head_dim=128,
        linear_num_key_heads=8,
        linear_num_value_heads=16,
        # hybrid layout: every `full_attention_interval`-th layer is full attention
        full_attention_interval=4,
        layer_types=None,
        # nanochat training extras
        logit_softcap=15.0,
        logit_softcap_end=100.0,
        logit_softcap_anneal_steps=2000,
        mtp_num_heads=0,
        mtp_weight=0.0,
        pad_token_id=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        # GatedDeltaNet
        self.linear_conv_kernel_dim = linear_conv_kernel_dim
        self.linear_key_head_dim = linear_key_head_dim
        self.linear_value_head_dim = linear_value_head_dim
        self.linear_num_key_heads = linear_num_key_heads
        self.linear_num_value_heads = linear_num_value_heads

        # RoPE: modeling code reads `config.rope_parameters` (a dict)
        if rope_parameters is None:
            rope_parameters = {
                "rope_type": "default",
                "rope_theta": rope_theta,
                "partial_rotary_factor": partial_rotary_factor,
            }
            if mrope_section is not None:
                rope_parameters["mrope_section"] = mrope_section
        rope_parameters.setdefault("rope_type", "default")
        rope_parameters.setdefault("rope_theta", rope_theta)
        rope_parameters.setdefault("partial_rotary_factor", partial_rotary_factor)
        self.rope_parameters = rope_parameters
        self.rope_theta = rope_parameters["rope_theta"]
        self.partial_rotary_factor = rope_parameters["partial_rotary_factor"]

        # Build the hybrid layer layout if not explicitly provided.
        self.full_attention_interval = full_attention_interval
        if layer_types is None:
            layer_types = [
                "full_attention" if (i + 1) % full_attention_interval == 0 else "linear_attention"
                for i in range(num_hidden_layers)
            ]
        self.layer_types = layer_types

        # nanochat training extras (read via getattr in the model)
        self.logit_softcap = logit_softcap
        self.logit_softcap_end = logit_softcap_end
        self.logit_softcap_anneal_steps = logit_softcap_anneal_steps
        self.mtp_num_heads = mtp_num_heads
        self.mtp_weight = mtp_weight

        super().__init__(
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


# Backwards-compatible aliases (the modeling file references these names).
Qwen3_5Config = Qwen3_5TextConfig


__all__ = ["Qwen3_5TextConfig", "Qwen3_5Config"]


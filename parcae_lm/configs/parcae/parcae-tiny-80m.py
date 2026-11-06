"""Tiny ~80M Parcae model configuration with diagonal injection."""

from parcae_lm.registry import DEFAULT_HF_ORG

configs = [
    dict(
        name="parcae-tiny-80m",
        hf_config=dict(org=DEFAULT_HF_ORG, name="parcae-tiny-80m"),
        # Context
        block_size=2048,
        vocab_size=32768,
        padding_multiple=64,
        n_embd=512,
        num_attention_heads=4,
        num_key_value_heads=4,
        intermediate_size=2048,
        bias=False,
        tie_embeddings=True,
        # Model class
        architecture_class_name="Parcae",
        block_class_name="TransformerPreNormBlock",
        norm_class_name="RMSNorm",
        norm_eps=1e-5,
        mlp_class_name="BaseMLP",
        nonlin_name="ReLU2",
        qk_norm=True,
        logit_softcap=None,
        prelude_norm=True,
        # Initialization
        init_strategy="scaled-zero",
        init_orthogonal=True,
        # Recurrent settings - diagonal injection
        injection_type="diagonal",
        state_init="like-init",
        recurrent_embedding_dimension=512,
        recurrent_intermediation_embedding_dimension=2048,
        n_layers_in_recurrent_block=2,
        n_layers_in_prelude=2,
        n_layers_in_coda=2,
        mean_recurrence=8,
        sampling_scheme="poisson-truncated-full",
        mean_backprop_depth=4,
        recurrent_iteration_method="per-sequence",
    ),
]

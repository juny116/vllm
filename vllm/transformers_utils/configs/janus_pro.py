# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Janus Pro configuration."""

from transformers import PretrainedConfig


class JanusProConfig(PretrainedConfig):
    """
    Configuration class for Janus Pro models.
    
    Janus Pro is a unified multimodal foundation model with vision understanding
    and generation capabilities. This config handles the multi_modality model type.
    """

    model_type = "multi_modality"

    def __init__(
        self,
        vision_config=None,
        language_config=None,
        text_config=None,
        use_image_id=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Janus Pro has separate vision and language configs
        if vision_config is None:
            vision_config = {}
        if language_config is None and text_config is None:
            language_config = {}
        elif text_config is not None:
            language_config = text_config

        # Convert dict to config objects if needed
        if isinstance(vision_config, dict):
            from transformers import CONFIG_MAPPING
            
            vision_model_type = vision_config.get("model_type", "siglip_vision_model")
            if vision_model_type in CONFIG_MAPPING:
                vision_config = CONFIG_MAPPING[vision_model_type](**vision_config)
            else:
                # Create a basic config
                vision_config = PretrainedConfig(**vision_config)

        if isinstance(language_config, dict):
            from transformers import CONFIG_MAPPING
            
            language_model_type = language_config.get("model_type", "qwen2")
            if language_model_type in CONFIG_MAPPING:
                language_config = CONFIG_MAPPING[language_model_type](**language_config)
            else:
                language_config = PretrainedConfig(**language_config)

        self.vision_config = vision_config
        self.language_config = language_config
        self.text_config = language_config  # Alias for compatibility
        self.use_image_id = use_image_id

        # Set architectures if not present
        if not hasattr(self, "architectures") or self.architectures is None:
            self.architectures = ["JanusProForConditionalGeneration"]

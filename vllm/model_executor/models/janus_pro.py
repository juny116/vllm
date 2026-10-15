# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Inference-only Janus-Pro model (text-only wrapper)."""

from typing import Iterable, List, Optional

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsLoRA, SupportsPP
from .utils import maybe_prefix

logger = init_logger(__name__)


class JanusProForCausalLM(nn.Module, SupportsPP, SupportsLoRA):
    """
    Janus Pro text-only wrapper.
    
    Delegates to vLLM's native Qwen2/Llama implementation,
    loading only language_model weights from Janus checkpoint.
    """
    
    # Delegate packed_modules_mapping to the underlying model
    @property
    def packed_modules_mapping(self):
        return getattr(self.model, 'packed_modules_mapping', {})
    
    # Delegate embedding_modules to the underlying model
    @property
    def embedding_modules(self):
        return getattr(self.model, 'embedding_modules', {})
    
    # Delegate embedding_padding_modules to the underlying model
    @property
    def embedding_padding_modules(self):
        return getattr(self.model, 'embedding_padding_modules', [])

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_config
        self.config = config

        # Get language model config
        if hasattr(config, 'language_config'):
            language_config = config.language_config
        elif hasattr(config, 'text_config'):
            language_config = config.text_config
        else:
            raise ValueError("Cannot find language_config or text_config in model config")
        
        # Determine the architecture
        language_arch = getattr(language_config, 'architectures', None)
        if language_arch is None:
            model_type = getattr(language_config, 'model_type', 'qwen2')
            if model_type == 'qwen2':
                language_arch = 'Qwen2ForCausalLM'
            elif model_type == 'llama':
                language_arch = 'LlamaForCausalLM'
            else:
                raise ValueError(f"Unsupported language model type: {model_type}")
        elif isinstance(language_arch, list):
            language_arch = language_arch[0]
        
        # Import the corresponding model class directly
        if language_arch == 'Qwen2ForCausalLM':
            from .qwen2 import Qwen2ForCausalLM
            model_cls = Qwen2ForCausalLM
        elif language_arch == 'LlamaForCausalLM':
            from .llama import LlamaForCausalLM
            model_cls = LlamaForCausalLM
        else:
            raise ValueError(f"Unsupported language architecture: {language_arch}")
        
        # Temporarily replace the hf_config with language_config
        # This is safe because we're inside __init__ and haven't exposed the object yet
        original_hf_config = vllm_config.model_config.hf_config
        vllm_config.model_config.hf_config = language_config
        
        # Initialize the language model WITHOUT prefix
        # We'll handle prefix in load_weights
        self.model = model_cls(vllm_config=vllm_config, prefix="")
        self._prefix = prefix  # Store for load_weights
        
        # Restore original config
        vllm_config.model_config.hf_config = original_hf_config
        
        logger.info(f"Janus Pro initialized with {language_arch} as language model (prefix='{prefix}')")
        
        # Debug: Print what parameters THIS wrapper expects
        all_param_names = [name for name, _ in self.named_parameters()]
        logger.info(f"JanusProForCausalLM (wrapper) has {len(all_param_names)} parameters")
        logger.info(f"Sample JanusProForCausalLM param names: {all_param_names[:5]}")
        
        # Debug: Print what parameters the underlying model expects
        model_param_names = [name for name, _ in self.model.named_parameters()]
        logger.info(f"Underlying {language_arch} has {len(model_param_names)} parameters")
        logger.info(f"Sample underlying model param names: {model_param_names[:5]}")

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get input embeddings."""
        return self.model.get_input_embeddings(input_ids)
    
    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Apply token embeddings to input_ids."""
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass through language model."""
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """Compute logits from hidden states."""
        return self.model.compute_logits(hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load model weights.
        
        Filters out vision-related weights and loads only language model weights.
        
        Checkpoint structure (Janus Pro):
          - language_model.model.embed_tokens.weight
          - language_model.model.layers.0.self_attn.q_proj.weight
          - language_model.lm_head.weight
        
        JanusProForCausalLM expects (because self.model adds 'model.' prefix):
          - model.model.embed_tokens.weight
          - model.model.layers.0.self_attn.qkv_proj.weight (merged)
          - model.lm_head.weight
        
        So we need to:
          1. Remove 'language_model.' -> 'model.embed_tokens.weight'
          2. Add 'model.' prefix -> 'model.model.embed_tokens.weight'
        """
        # Convert to list to allow filtering
        weights_list = list(weights)
        
        # Filter weights - only keep language_model weights
        language_weights = []
        for name, tensor in weights_list:
            # Skip vision-related weights
            if any(prefix in name for prefix in ['vision_model', 'vision_encoder', 
                                                   'vision_projector', 'aligner',
                                                   'gen_vision_model', 'gen_aligner',
                                                   'gen_head', 'gen_embed']):
                continue
            
            # Keep only language_model weights
            if not name.startswith('language_model.'):
                continue
            
            # Remove 'language_model.' prefix
            # 'language_model.model.layers.0.weight' -> 'model.layers.0.weight'
            new_name = name[len('language_model.'):]
            
            # Add 'model.' prefix because self.model is registered as a submodule
            # 'model.layers.0.weight' -> 'model.model.layers.0.weight'
            # 'lm_head.weight' -> 'model.lm_head.weight'
            new_name = f"model.{new_name}"
            
            # Add our prefix if we have one (shouldn't happen)
            if self._prefix:
                new_name = f"{self._prefix}.{new_name}"
            
            language_weights.append((new_name, tensor))
        
        logger.info(f"Loading {len(language_weights)} language model weights (prefix='{self._prefix}')")
        if language_weights:
            logger.info(f"Sample converted names: {[name for name, _ in language_weights[:5]]}")
        
        # Use AutoWeightsLoader to handle qkv merging and other transformations
        from .utils import AutoWeightsLoader
        
        loader = AutoWeightsLoader(self)
        return loader.load_weights(language_weights)

    def make_empty_intermediate_tensors(
        self, batch_size: int, dtype: torch.dtype, device: torch.device
    ) -> IntermediateTensors:
        """Create empty intermediate tensors."""
        return self.model.make_empty_intermediate_tensors(batch_size, dtype, device)
    
    # Delegate any other methods that might be needed
    def __getattr__(self, name: str):
        """Delegate attribute access to the underlying model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            # If not found in this class, try the underlying model
            return getattr(self.model, name)

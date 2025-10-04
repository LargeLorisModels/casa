import os
import json
from pathlib import Path
from typing import Optional, List

import torch
from transformers import (
	AutoModelForCausalLM,
	AutoTokenizer,
	PreTrainedModel,
	PreTrainedTokenizer,
)


class LLM:
	"""Wrapper for language models with support for constrained generation.
	
	Attributes:
		model_id: Hugging Face model identifier.
		model: The loaded transformer model.
		is_chat_model: Enables chat template for instruct models (default:true)
		tokenizer: The associated tokenizer.
		device: Device where the model is loaded.
	"""
	
	def __init__(
		self,
		model: PreTrainedModel,
		tokenizer: PreTrainedTokenizer,
		model_id: str,
		is_chat_model: bool = True
	):
		"""Initialize LLM wrapper.
		
		Args:
			model: Pre-loaded transformer model.
			tokenizer: Pre-loaded tokenizer.
			model_id: Model identifier string.
			is_chat_model: Configures chat template.
		"""
		self.model = model
		self.tokenizer = tokenizer
		self.model_id = model_id
		self.device = model.device
		self.is_chat_model = is_chat_model
		
		# Ensure tokenizer has pad token
		if self.tokenizer.pad_token is None:
			self.tokenizer.pad_token = self.tokenizer.eos_token
	
	@classmethod
	def from_pretrained(
		cls,
		model_id: str,
		is_chat_model: bool = True,
		dtype: torch.dtype = torch.bfloat16,
		device_map: str = "auto",
		hf_token: Optional[str] = None,
		**model_kwargs,
	) -> 'LLM':
		"""Load a language model from Hugging Face.
		
		Args:
			model_id: Hugging Face model identifier.
			dtype: Data type for model weights.
			device_map: Device mapping strategy for model loading.
			hf_token: Hugging Face API token for gated models.
			**model_kwargs: Additional arguments passed to AutoModelForCausalLM.
			
		Returns:
			Initialized LLM instance.
		"""
		# Handle HF token
		if hf_token is not None:
			os.environ["HF_TOKEN"] = hf_token
		elif "HF_TOKEN" not in os.environ:
			secrets_path = Path("secrets.json")
			if secrets_path.exists():
				with open(secrets_path) as f:
					secrets = json.load(f)
					if "HF_TOKEN" in secrets and secrets["HF_TOKEN"] != "your_token":
						os.environ["HF_TOKEN"] = secrets["HF_TOKEN"]
		
		# Load tokenizer
		tokenizer = AutoTokenizer.from_pretrained(model_id)
		
		# Load model
		model = AutoModelForCausalLM.from_pretrained(
			model_id,
			device_map=device_map,
			dtype=dtype,
			**model_kwargs,
		)
		model.eval()
		
		return cls(model=model, tokenizer=tokenizer, model_id=model_id, is_chat_model=is_chat_model)
	
	def format_prompt(self, prompt: str) -> str:
		"""Format prompt based on model type (chat vs base).
		
		Args:
			prompt: Raw prompt string.
			
		Returns:
			Formatted prompt string.
		"""

		if self.is_chat_model:
			messages = [{"role": "user", "content": prompt}]
			formatted = self.tokenizer.apply_chat_template(
				messages,
				tokenize=False,
				add_generation_prompt=True,
			)
			assert isinstance(formatted, str)
			return formatted
		else:
			return prompt
	
	def encode(self, text: str, return_tensors: str = "pt") -> torch.Tensor:
		"""Tokenize text.
		
		Args:
			text: Input text to tokenize.
			return_tensors: Format for returned tensors.
			
		Returns:
			Tokenized input IDs.
		"""
		return self.tokenizer.encode(text, return_tensors=return_tensors)
	
	def decode(self, token_ids: torch.Tensor) -> str:
		"""Decode token IDs to text.
		
		Args:
			token_ids: Token IDs to decode.
			
		Returns:
			Decoded text string.
		"""
		return self.tokenizer.decode(token_ids, skip_special_tokens=False)
	
	def batch_decode(self, token_ids: torch.Tensor) -> List[str]:
		"""Decode batch of token IDs to text.
		
		Args:
			token_ids: Batch of token IDs to decode.
			
		Returns:
			List of decoded text strings.
		"""
		return self.tokenizer.batch_decode(token_ids, skip_special_tokens=False)
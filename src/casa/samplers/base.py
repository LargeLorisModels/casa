from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import torch
from casa.llm import LLM
from casa.grammar import Grammar

@dataclass
class SamplingResult:
    """Result from a sampling operation.
    
    Attributes:
        tokens: List of token strings.
        token_ids: List of token IDs.
        text: Decoded text.
        raw_logprob: Raw log probability from the model.
        constrained_logprob: Log probability under grammar constraints.
        success: Whether the sample satisfied constraints.
    """
    tokens: List[str]
    token_ids: List[int]
    text: str
    raw_logprob: float
    constrained_logprob: Optional[float] = None
    success: bool = True
    attempts: int = 1


class BaseSampler(ABC):
    """Abstract base class for sampling algorithms.
    
    All samplers should inherit from this class and implement the sample() method.
    """
    
    def __init__(
        self,
        llm,
        grammar,
        max_new_tokens: int = 512,
    ):
        """Initialize base sampler.
        
        Args:
            llm: LLM instance.
            grammar: Grammar instance.
            max_new_tokens: Maximum number of tokens to generate.
        """
        
        if not isinstance(llm, LLM):
            raise TypeError(f"llm must be an LLM instance, got {type(llm)}")
        if not isinstance(grammar, Grammar):
            raise TypeError(f"grammar must be a Grammar instance, got {type(grammar)}")
        
        self.llm = llm
        self.grammar = grammar
        self.max_new_tokens = max_new_tokens
    
    @abstractmethod
    def sample(
        self,
        prompt: str,
        n_samples: int = 1,
        **kwargs,
    ) -> List[SamplingResult]:
        """Generate samples from the model.
        
        Args:
            prompt: Input prompt text.
            n_samples: Number of samples to generate.
            **kwargs: Additional sampler-specific arguments.
            
        Returns:
            List of sampling results.
        """
        pass
    
    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        """Encode and format prompt.
        
        Args:
            prompt: Raw prompt string.
            
        Returns:
            Encoded prompt tensor on model device.
        """
        formatted_prompt = self.llm.format_prompt(prompt)
        prompt_ids = self.llm.tokenizer.encode(
            formatted_prompt,
            return_tensors="pt",
            add_special_tokens=False,
        ).to(self.llm.device)
        return prompt_ids
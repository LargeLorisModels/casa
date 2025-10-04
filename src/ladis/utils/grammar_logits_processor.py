"""Grammar-constrained logits processor for MCMC sampling."""

import torch
import xgrammar
from transformers.generation.logits_process import LogitsProcessor


class GrammarLogitsProcessor(LogitsProcessor):
    """Logits processor that enforces grammar constraints during generation.
    
    This processor applies grammar-based token filtering at each generation step,
    ensuring that only valid tokens according to the grammar are sampled.
    
    Args:
        tokenizer: The tokenizer associated with the model.
        grammar_constraint: Grammar constraint object from llguidance.
        device: Device to run computations on.
        prompt_length: Length of the prompt (to track generation start).
    """
    
    def __init__(
        self,
        tokenizer,
        grammar_constraint,
        device: torch.device,
        prompt_length: int,
    ):
        """Initialize grammar-constrained processor.
        
        Args:
            tokenizer: Tokenizer for the language model.
            grammar_constraint: Grammar recognizer instance.
            device: Device for tensor operations.
            prompt_length: Length of the prompt in tokens.
        """
        self.tokenizer = tokenizer
        self.grammar_constraint = grammar_constraint
        self.device = device
        self.prompt_length = prompt_length
    
    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Apply grammar constraints to logits.
        
        Args:
            input_ids: Input token IDs of shape (batch_size, sequence_length).
            scores: Raw logits of shape (batch_size, vocab_size).
            
        Returns:
            Constrained logits with invalid tokens masked out.
            
        Raises:
            ValueError: If the sequence violates grammar constraints.
        """
        # Extract generated tokens (excluding prompt)
        generated_tokens = self._get_generated_tokens(input_ids)
        
        # Advance grammar parser
        if not self.grammar_constraint.try_advance_token_ids(generated_tokens):
            raise ValueError(
                f"Grammar constraint violated at tokens: {generated_tokens}"
            )
        
        # Get valid tokens according to grammar
        acceptance = self.grammar_constraint.filter_vocab()
        
        # Apply token mask to scores
        scores = scores.clone()
        xgrammar.apply_token_bitmask_inplace(
            scores,
            acceptance.to(scores.device, non_blocking=True)
        )
        
        # Mask out tokens beyond vocabulary size
        scores[0, self.grammar_constraint.ll_tokenizer.vocab_size:] = float('-inf')
        
        return scores
    
    def _get_generated_tokens(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """Extract generated tokens from input_ids.
        
        Args:
            input_ids: Full input token IDs including prompt.
            
        Returns:
            Generated tokens only (excluding prompt).
        """
        assert input_ids.shape[0] == 1, "Batch size must be 1"
        return input_ids[0, self.prompt_length:]
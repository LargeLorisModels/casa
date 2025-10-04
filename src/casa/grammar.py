from typing import Optional
from transformers import PreTrainedTokenizer

from casa.utils.llguidance_recognizer import LlguidanceTokenRecognizer


class Grammar:
    """Grammar constraint for structured generation.
    
    Attributes:
        grammar_str: Grammar specification string.
        recognizer: Underlying grammar recognizer.
    """
    
    def __init__(self, grammar_str: str, tokenizer: PreTrainedTokenizer):
        """Initialize grammar constraint.
        
        Args:
            grammar_str: Grammar specification in EBNF or Lark format.
            tokenizer: Tokenizer associated with the language model.
        """
        self.grammar_str = grammar_str
        self.recognizer = LlguidanceTokenRecognizer(grammar_str, tokenizer)
    
    @classmethod
    def from_file(cls, path: str, tokenizer: PreTrainedTokenizer) -> 'Grammar':
        """Load grammar from file.
        
        Args:
            path: Path to grammar file (.ebnf or .lark).
            tokenizer: Tokenizer associated with the language model.
            
        Returns:
            Grammar instance.
        """
        with open(path, 'r') as f:
            grammar_str = f.read()
        return cls(grammar_str, tokenizer)
    
    @classmethod
    def from_string(cls, grammar_str: str, tokenizer: PreTrainedTokenizer) -> 'Grammar':
        """Create grammar from string.
        
        Args:
            grammar_str: Grammar specification string.
            tokenizer: Tokenizer associated with the language model.
            
        Returns:
            Grammar instance.
        """
        return cls(grammar_str, tokenizer)
    
    def reset(self) -> None:
        """Reset the grammar recognizer state."""
        self.recognizer.reset()
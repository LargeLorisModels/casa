import torch
from typing import List, Tuple


def get_seq_logprob_from_scores(
    scores: torch.Tensor,
    query_ids: torch.Tensor,
    eos_token_id: int,
) -> torch.Tensor:
    """Calculate log probability of sequences from model scores.
    
    Args:
        scores: Model scores of shape (batch_size, seq_len, vocab_size).
        query_ids: Token IDs of shape (batch_size, seq_len).
        eos_token_id: EOS token ID for determining sequence end.
        
    Returns:
        Log probabilities of shape (batch_size,).
    """
    assert scores.shape[0] == query_ids.shape[0], "Batch sizes must match"
    assert scores.shape[1] == query_ids.shape[1], "Sequence lengths must match"
    
    # Convert to log probabilities
    logprobs = torch.log_softmax(scores, dim=-1)
    
    batch_size, seq_len = query_ids.shape
    result = torch.zeros(batch_size, device=scores.device)
    
    # Calculate log probability for each sequence
    for i in range(batch_size):
        # Get log probabilities for this sequence's tokens
        seq_logprobs = logprobs[i, torch.arange(seq_len), query_ids[i]]
        
        # Find first EOS token position
        eos_mask = query_ids[i] == eos_token_id
        eos_positions = torch.nonzero(eos_mask)
        
        if eos_positions.shape[0] > 0:
            # Sum up to and including first EOS token
            first_eos_pos = eos_positions[0].item()
            result[i] = seq_logprobs[:first_eos_pos + 1].sum()
        else:
            # No EOS token, sum all logprobs
            result[i] = seq_logprobs.sum()
    
    return result


def unbatch_sequences(
    sequences: torch.Tensor,
    eos_token_id: int,
) -> List[torch.Tensor]:
    """Unbatch sequences and remove padding.
    
    Args:
        sequences: Batched sequences of shape (batch_size, seq_len).
        eos_token_id: EOS token ID for detecting sequence end.
        
    Returns:
        List of unbatched sequences with padding removed.
    """
    result = []
    for i in range(sequences.shape[0]):
        # Find first EOS token
        eos_mask = sequences[i] == eos_token_id
        eos_idx = torch.nonzero(eos_mask)
        
        if eos_idx.shape[0] > 0:
            # Keep up to and including first EOS
            eos_idx = eos_idx[0].item()
            result.append(sequences[i][:eos_idx + 1])
        else:
            # No EOS, keep full sequence
            result.append(sequences[i])
    
    return result


def scores_to_top_k(
    scores: torch.Tensor,
    k: int,
) -> List[List[Tuple[int, float]]]:
    """Extract top-k tokens and their log probabilities from scores.
    
    Args:
        scores: Model scores of shape (seq_len, vocab_size) or (batch_size, seq_len, vocab_size).
        k: Number of top tokens to return.
        
    Returns:
        List of (token_id, log_prob) tuples for each position.
    """
    result = []
    for step_scores in scores:
        log_probs = torch.log_softmax(step_scores, dim=-1)
        top_probs, top_token_ids = torch.topk(log_probs, k=k)
        top_choices = list(zip(top_token_ids.tolist(), top_probs.tolist()))
        result.append(top_choices)
    return result
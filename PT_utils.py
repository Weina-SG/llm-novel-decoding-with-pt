import torch
import torch.nn.functional as F

def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.

    Args:
        logits (Tensor): logits distribution shape (batch_size, vocab_size).
        top_k (int): keep only top k tokens with highest probability (default: 0, disabled).
        top_p (float): keep the top tokens with cumulative probability >= top_p (default: 1.0, disabled).
        filter_value (float): value to replace filtered logits with (default: -Inf).
        min_tokens_to_keep (int): ensure at least this many tokens are kept (default: 1).

    Returns:
        Tensor: logits tensor with filtered values set to filter_value.
    """
    # Safety checks
    if top_k > 0:
        # Only keep top_k logits per batch
        top_k = max(top_k, min_tokens_to_keep)
        # Remove tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least min_tokens_to_keep tokens
        if min_tokens_to_keep > 1:
            sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
        # Shift right to keep first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        # Scatter to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value

    return logits

import json
import torch
import numpy as np
from nltk import ngrams
from transformers import GPT2LMHeadModel, GPT2Tokenizer

import warnings
warnings.filterwarnings("ignore", message="`loss_type=None` was set in the config but it is unrecognised")

# Device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model & Tokenizer Setup
MODEL_NAME = "gpt2-medium"
GPT2_TOKENIZER = GPT2Tokenizer.from_pretrained(MODEL_NAME)
if GPT2_TOKENIZER.pad_token_id is None:
    GPT2_TOKENIZER.pad_token = GPT2_TOKENIZER.eos_token
GPT2_MODEL = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
GPT2_MODEL.eval()

def distinct_n(outputs, n=2):
    """Calculate Distinct-N metric"""
    total_ngrams = 0
    unique_ngrams = set()
    for text in outputs:
        if not isinstance(text, str):
            continue
        tokens = text.split()
        if len(tokens) < n:
            continue
        text_ngrams = list(ngrams(tokens, n))
        total_ngrams += len(text_ngrams)
        unique_ngrams.update(text_ngrams)
    return len(unique_ngrams) / total_ngrams if total_ngrams > 0 else 0

def perplexity(texts):
    """Calculate perplexity using GPT-2"""
    ppl_scores = []
    for text in texts:
        inputs = GPT2_TOKENIZER.encode(text, return_tensors="pt", truncation=True, max_length=1024).to(DEVICE)
        with torch.no_grad():
            outputs = GPT2_MODEL(inputs, labels=inputs)
            loss = outputs.loss
        ppl = torch.exp(loss).item()
        ppl_scores.append(ppl)
    return np.mean(ppl_scores) if ppl_scores else None

def evaluate(outputs):
    """Evaluate Distinct-1, Distinct-2, Distinct-3, and Perplexity"""
    metrics = {
        'distinct_1': distinct_n(outputs, 1),
        'distinct_2': distinct_n(outputs, 2),
        'distinct_3': distinct_n(outputs, 3),
        'perplexity': perplexity(outputs)
    }
    return metrics

def main():
    input_json = "generated_outputs.json"
    output_json = "diversity_perplexity_results.json"
    with open(input_json) as f:
        eval_data = json.load(f)

    results = []
    for item in eval_data:
        prompt_id = item.get("id", "")
        outputs = item.get("generated_text", [])
        metrics = evaluate(outputs)
        results.append({
            "prompt_id": prompt_id,
            "metrics": metrics
        })

    with open(output_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_json}")

if __name__ == "__main__":
    main()
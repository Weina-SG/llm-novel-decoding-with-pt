import sys
import json
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer#, DynamicCache
from PT_decode import parallel_tempering

# ===== Configuration =====
MODEL_NAME = "gpt2-medium"  # Use medium for better performance
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T_a = 1e-5  # Cold chain temperature
T_b = 2.0  # Hot chain temperature
MAX_STEPS = 200  # Total generation steps
SWAP_INTERVAL = 5  # Steps between swap attempts
TOP_K = 50  # Top-k sampling
TOP_P = 0.95  # Top-p (nucleus) sampling
PROMPT_FILE = "input_data.json"  # Input dataset file
OUTPUT_FILE = "generated_outputs.json"  # Output file for generated text

# ===== Model & Tokenizer Setup =====
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
# Ensure pad_token is set
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
# tokenizer.pad_token = tokenizer.eos_token  # Critical for GPT-2
model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()
print(f"Loaded {MODEL_NAME} on {DEVICE}")
print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
# print(model.config)

# ===== Load Dataset =====
with open(PROMPT_FILE, "r") as file:
    dataset = json.load(file)

print(f"Loaded {len(dataset)} prompts from {PROMPT_FILE}")

# ===== Text Generation =====
generated_data = []

for ind, entry in enumerate(dataset):
    # Limit to first n entries for testing
    if ind>=100:
        break

    # Print progress
    print(f"\nProcessing entry {ind+1}/{len(dataset)}: {entry['id']}")

    # Check if 'prompt' key exists
    if 'prompt' in entry:
        PROMPT = entry['prompt']  # Extract prompt text
        PROMPT = PROMPT.split(":cite[")[0].strip()
    else:
        continue

    # Initialize chains with the same prompt
    outputA, outputB = parallel_tempering(prompt=PROMPT, 
                                          model=model, 
                                          tokenizer=tokenizer, 
                                          device=DEVICE, 
                                          Ta=T_a, 
                                          Tb=T_b, 
                                          max_steps=MAX_STEPS,
                                          topk=TOP_K, 
                                          topp=TOP_P)
    # outputA, outputB = '', '' #fake the output

    # Encode the prompt
    INPUT_ids = tokenizer.encode(PROMPT, return_tensors="pt").to(DEVICE)
    
    # Create attention mask
    attention_mask = (INPUT_ids != tokenizer.pad_token_id).long()

    # ===== T=T_a, generate output ===== 

    # Generate text
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=INPUT_ids,
            do_sample=True,  # Must be True for top_p to take effect
            attention_mask=attention_mask,  # Fix attention mask warning
            max_new_tokens=MAX_STEPS,
            temperature=T_a,
            top_k=TOP_K,
            top_p=TOP_P,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode the generated text
    Ta_generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # ===== T=T_b, generate output ===== 
    with torch.no_grad():
        output_ids = model.generate(
            INPUT_ids,
            do_sample=True,  # Must be True for top_p to take effect
            attention_mask=attention_mask,  # Add attention mask here
            max_new_tokens=MAX_STEPS,
            temperature=T_b,
            top_k=TOP_K,
            top_p=TOP_P,
            pad_token_id=tokenizer.eos_token_id
        )

    # Decode the generated text
    Tb_generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Store the result
    generated_text = [outputA, outputB, Ta_generated_text, Tb_generated_text]
    generated_data.append({
        "id": entry['id'],
        "prompt": PROMPT,
        "task": entry['task'],
        "generated_text": generated_text
    })

    print(f"Processed prompt: {PROMPT}")
    print(f"Generated text: {' ||| '.join(generated_text)}\nEnd of Generated text\n")

# ===== Save Results =====

# Save generated_data to a JSON file
with open(OUTPUT_FILE, "w") as file:
    json.dump(generated_data, file, indent=4)

print(f"Generated data has been saved to {OUTPUT_FILE}")
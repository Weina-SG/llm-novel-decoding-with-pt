import torch
import torch.nn.functional as F
from PT_utils import top_k_top_p_filtering
# from transformers import GPT2LMHeadModel, GPT2Tokenizer

def parallel_tempering(prompt, model, tokenizer, device, Ta=1.0, Tb=2.0, max_steps=50, topk=50, topp=0.95):
    """
    Perform parallel tempering with two chains at different temperatures.
    
    Args:
        Ta (float): Temperature for chain A (cold chain).
        Tb (float): Temperature for chain B (hot chain).
        max_steps (int): Total number of generation steps.
    """
    # Initialize device and model
    model.eval()
    print("model prepared")

    # Softmax temperatures for the two chains
    T_a = Ta  # cold chain
    T_b = Tb  # hot chain

    # Initialize both chains with the same prompt
    # prompt = "Once upon a time, "
    chainA_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    chainB_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    chainA_past = None
    chainB_past = None

    # Cumulative "energy" (negative log-prob) of each chain
    chainA_energy = 0.0
    chainB_energy = 0.0

    for step in range(max_steps):
        print("step:", step)
        # ---- Chain A (temperature T_a) ----
        with torch.no_grad():
            if step == 0:
                outputsA = model(input_ids=chainA_ids.to(device), use_cache=True)
            else:
                next_input_A = chainA_ids[:, -1].unsqueeze(-1).to(device)
                outputsA = model(input_ids=next_input_A, past_key_values=chainA_past, use_cache=True)

        # outputsA = model(chainA_ids, past_key_values=chainA_past, use_cache=True)
        logitsA = outputsA.logits[:, -1, :]
        filtered_logitsA = top_k_top_p_filtering(logitsA, top_k=topk, top_p=topp)

        # Apply temperature by scaling logits (dividing by T)
        log_probsA = F.log_softmax(filtered_logitsA / T_a, dim=-1)
        # Sample next token (you could also do argmax for deterministic behavior)
        next_token_A = torch.multinomial(torch.exp(log_probsA), num_samples=1)
        # print("next_token_A:", next_token_A)
        # Update chainA energy: negative log-prob of chosen token
        chainA_energy += -log_probsA[0, next_token_A]
        # Update tokens and past state
        chainA_ids = torch.cat([chainA_ids, next_token_A], dim=-1)
        chainA_past = outputsA.past_key_values
        # print("chainA_energy:", chainA_energy)
        # print("chainA_ids:", chainA_ids)
        #print("chainA_past:", chainA_past) # key values, output too long

        # ---- Chain B (temperature T_b) ----
        with torch.no_grad():
            if step == 0:
                outputsB = model(input_ids=chainB_ids.to(device), use_cache=True)
            else:
                next_input_B = chainB_ids[:, -1].unsqueeze(-1).to(device)
                outputsB = model(input_ids=next_input_B, past_key_values=chainB_past, use_cache=True)
        # outputsB = model(chainB_ids, past_key_values=chainB_past, use_cache=True)
        logitsB = outputsB.logits[:, -1, :]
        filtered_logitsB = top_k_top_p_filtering(logitsB, top_k=topk, top_p=topp)
        log_probsB = F.log_softmax(filtered_logitsB / T_b, dim=-1)
        next_token_B = torch.multinomial(torch.exp(log_probsB), num_samples=1)
        # print("next_token_B:", next_token_B)
        chainB_energy += -log_probsB[0, next_token_B]
        chainB_ids = torch.cat([chainB_ids, next_token_B], dim=-1)
        chainB_past = outputsB.past_key_values
        # print("chainB_energy:", chainB_energy)
        # print("chainB_ids:", chainB_ids)
        #print("chainB_past:", chainB_past) # key values, output too long

        # ---- Metropolis Swap ----
        # Every few steps, attempt an exchange of states
        if step % 5 == 0 and step > 0:
            # Calculate Metropolis acceptance probability
            # Here, E = negative log-likelihood, so use (1/T_i - 1/T_j)*(E_j - E_i)
            delta = (1.0/T_a - 1.0/T_b) * (chainB_energy - chainA_energy)
            # Avoid numerical issues
            try:
                acceptance = torch.exp(delta).item()
            except OverflowError:
                acceptance = float('inf') if delta > 0 else 0.0
            # Swap with probability min(1, exp(delta))
            if torch.rand(1).item() < min(1.0, acceptance):
                # Swap the entire state (token ids and past_key_values)
                temp_ids, temp_past, temp_energy = chainA_ids, chainA_past, chainA_energy
                chainA_ids, chainA_past, chainA_energy = chainB_ids, chainB_past, chainB_energy
                chainB_ids, chainB_past, chainB_energy = temp_ids, temp_past, temp_energy
                print(f"--- Swap accepted at step {step}! ---")

    # Decode and print the final outputs
    outputA = tokenizer.decode(chainA_ids[0], skip_special_tokens=True)
    outputB = tokenizer.decode(chainB_ids[0], skip_special_tokens=True)

    return outputA, outputB
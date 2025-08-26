# LLM Novel Decoding with Parallel Tempering

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
**A physics-inspired decoding algorithm that helps Large Language Models (LLMs) generate more creative and varied text without losing coherence.**

## Overview

**The Problem:** There's a fundamental trade-off in LLM decoding. Low-temperature sampling produces coherent but boring and repetitive text. High-temperature sampling is more creative but often leads to incoherent, nonsensical outputs ("hallucinations").

**Our Solution:** This repository implements **Parallel Tempering (PT)**, a decoding algorithm inspired by replica-exchange Monte Carlo methods from statistical physics. It runs two parallel sampling chains(useing T=1.0 and 2.0 as example, user can adjust based on own needs):
- A **"Cold" chain** (`T=1.0`) to preserve coherence.
- A **"Hot" chain** (`T>1.0`) to explore creative, low-probability tokens.

Periodically, the chains attempt to **swap their internal states** based on a Metropolis-Hastings criterion. This allows the cold chain to "inherit" promising creative discoveries from the hot chain without sacrificing its stability.

In effect, PT performs a more intelligent and efficient exploration of the model's probability distribution, leading to more engaging and diverse generations.

## Key Findings

Our experiments with GPT-2 Medium demonstrate that Parallel Tempering decoding effectively finds a "sweet spot" between standard sampling methods:

| Method | Perplexity (PPL) ↓ | Distinct-1 ↑ |
| :--- | :--- | :--- |
| Baseline (T=1.0) | 12.27 | 0.64 |
| Baseline (T=2.0) | 42.02 | 0.73 |
| **PT (Cold Chain)** | **16.86** | **0.67** |
| **PT (Hot Chain)** | **16.43** | **0.67** |
|

**Quantitative:** The PT-enhanced cold chain achieves higher diversity (Distinct-1) than standard T=1 sampling, while maintaining significantly lower perplexity (higher coherence) than standard T=2 sampling.

**Qualitative:** Generations are noticeably more varied and narratively interesting than standard sampling, without devolving into nonsense.
## How It Works (Theoretical Background)

The method is analogous to Parallel Tempering in computational physics. The "energy" of a generated sequence $\mathbf{x}$ is defined as its negative log-likelihood:

$$E(\mathbf{x}) = -\log P(\mathbf{x})$$

The core mechanism is the potential swap between the two chains. The probability of accepting a swap between the state of the cold chain $\mathbf{x}_c$ (at temperature $T_c$) and the hot chain $\mathbf{x}_h$ (at temperature $T_h$) is governed by the Metropolis-Hastings criterion:

$$
A = \min\left\{1,\, \exp\left[\left(\frac{1}{T_c} - \frac{1}{T_h}\right)\left(E(\mathbf{x}_c) - E(\mathbf{x}_h)\right)\right]\right\}
$$

This criterion ensures the detailed balance condition is maintained, allowing the cold chain to efficiently sample from a more diverse and creative distribution while retaining high coherence. The user's choice of $T_c$ and $T_h$ directly controls the balance between exploration (creativity) and exploitation (coherence).

# Getting Started with the repo

This repository contains scripts for generating text sequences using a language model(GPT-2 medium) and evaluating the generated outputs based on specific metrics(Distinct_1,2,3, perplexity).

## Files Overview

- **`PT_decode.py`**: Implements the parallel tempering decoding algorithm for improved text generation.
- **`PT_utils.py`**: Provides utility functions to contrain parameters in the decoding process.
- **`generation.py`**: The main script to run the text generation pipeline. It uses `input_data.json` as input and generates text sequences, saving the results to a JSON file.
- **`evaluation.py`**: Evaluates the generated text sequences using predefined metrics and outputs the evaluation scores.
- **`input_data.json`**: The input dataset containing prompts for text generation.

## Usage

### 1. Set Up a Virtual Environment (Optional but Recommended)
It is recommended to use a virtual environment to manage dependencies. To create and activate a virtual environment:

#### On Linux/MacOS:
```bash
python3 -m venv venv
source venv/bin/activate
```
#### On Windows:
```
python -m venv venv
venv\Scripts\activate
```
Once the virtual environment is activated, install the required dependencies.
### 2. Install Dependencies
Make sure to install the necessary libraries before running the scripts:
```
pip install -r requirements.txt
```
### 3. Generate Text Sequences
Run the `generation.py` script to generate text sequences:
```bash
python generation.py
```
- Input: input_data.json
- Output: A JSON file containing the generated text sequences.
### 4. Evaluate Generated Text
After generating the text, evaluate the outputs using the **`evaluation.py`** script:
```bash
python evaluation.py
```
- This script calculates evaluation metrics and provides scores for the generated outputs.

## Requirements
- Python 3.8+
- PyTorch
- Transformers library
- GPU (optional but recommended for faster processing)

## Notes
- Ensure that **`input_data.json`** is properly formatted as a JSON file containing prompts for generation.
- Modify the parameters in **`generation`**.py to customize the generation process.

## License
This project is licensed under the MIT License. You are free to use, modify, and distribute this code for research and commercial purposes, provided that proper attribution is given.

See the [LICENSE](LICENSE) file for details.

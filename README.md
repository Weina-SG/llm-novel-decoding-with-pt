# Text Generation and Evaluation Pipeline

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
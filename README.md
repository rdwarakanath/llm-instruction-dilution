# LLM Instruction Dilution

Empirical study of how prompt length and instruction placement affect
the output quality of local LLMs (3B and 8B parameter models) running
via Ollama.

## Setup

```bash
# 1. Create virtual environment (Python 3.10.11)
python -m venv venv
venv\Scripts\activate        # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install src as package (editable)
pip install -e .

# 4. Copy .env and fill in values
copy .env.example .env

# 5. Pull models
ollama pull llama3.2:3b
ollama pull llama3.1:8b
```

## Run Experiments

```bash
# Single config test run
python experiments/run_single_config.py --config experiments/configs/baseline.yaml

# Full experiment
python experiments/run_experiment.py --config experiments/configs/exp_prompt_length.yaml
```

## Project Structure

See FOLDER_STRUCTURE.md for full details.

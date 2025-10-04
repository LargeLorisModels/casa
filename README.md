![CASA](logo.png)

Constrained aligned sampling algorithms for language models via CARS, MCMC, and rejection sampling variants.

## Installation

### Prerequisites

- Python 3.12+
- CUDA-compatible GPU (recommended)

### Install from source

```bash
# Clone the repository
git clone https://github.com/LargeLorisModels/casa.git
cd casa

# Create virtual environment (optional but recommended)
python -m venv .venv
source .venv/bin/activate

# Install package in editable mode
pip install -e .
```

### Using uv (faster)

```bash
# Install uv if you haven't
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create environment and install
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Quick Start

```python
from casa import LLM, Grammar, CARS

llm = LLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

grammar_str = """
start: CHARACTER " " ACTION " " LOCATION "."
CHARACTER: "a dragon" | "a knight" | "a wizard"
ACTION: "discovered" | "protected" | "enchanted"
LOCATION: "the castle" | "the forest" | "the treasure"
"""

prompt = "Once upon a time,"
grammar = Grammar.from_string(grammar_str, llm.tokenizer)
sampler = CARS(llm, grammar, max_new_tokens=32, verbose=True)
results = sampler.sample(prompt, n_samples=10, max_attempts=100)

if results:
	print("\nGenerated samples,")
	for i, result in enumerate(results, 1):
		print(f"  {i}. {prompt} {result.text}")
else:
	print("Failed to generate any samples")
```

## Example Output

With `verbose=True`, you'll see the rejection samplers performance in real-time. Running above code,

```
Sample 01/10: ████████████████████████████████████████ 78 attempts
Sample 02/10: ████████ 12 attempts
Sample 03/10: █ 2 attempts
Sample 04/10: █ 1 attempts
Sample 05/10: █ 1 attempts
...

Generated samples:
  1. Once upon a time, a dragon enchanted the castle.
  2. Once upon a time, a dragon enchanted the forest.
  3. Once upon a time, a dragon enchanted the forest.
 ...
```

## Available Samplers

- **CARS**: Constrained Adaptive Rejection Sampling
- **MCMC**: Markov Chain Monte Carlo sampling. Avaliable variants,
  - _Uniform_ - Randomly resamples from any position. Balances exploration with structural preservation.
  - _Priority_ - Resample higher perplexity regions first. Targets uncertain tokens for refinement.
  - _Restart_ - Generates from scratch. Independent proposals via importance sampling.
- **ARS**: Adaptive Rejection Sampling
- **RSFT**: Rejection Sampling with First Token constraints
- **RS**: Basic Rejection Sampling

## Running Tests

```bash
# Run the example
python tests/test_cars.py

# Or other samplers
python tests/test_mcmc.py
```

## References

CASA implements the following algorithms from:

- **RS, ARS, RSFT, CARS**  
  _Constrained Adaptive Rejection Sampling_  
  Preprint | [arXiv:2510.01902](https://arxiv.org/abs/2510.01902)

- **MCMC - Uniform, Priority, Restart**  
  _Constrained Sampling for Language Models Should Be Easy: An MCMC Perspective_  
  NeurIPS 2026 | [arXiv:2506.05754](https://arxiv.org/abs/2506.05754)

## License

MIT License

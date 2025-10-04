from casa import LLM, Grammar, MCMC

llm = LLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")

grammar_str = """
start: CHARACTER " " ACTION " " LOCATION "."
CHARACTER: "a dragon" | "a knight" | "a wizard"
ACTION: "discovered" | "protected" | "enchanted"
LOCATION: "the castle" | "the forest" | "the treasure"
"""

prompt = "Once upon a time,"
grammar = Grammar.from_string(grammar_str, llm.tokenizer)

# Available variants - uniform, priority, restart
sampler = MCMC(llm, grammar, variant="restart", max_new_tokens=32)
results = sampler.sample(prompt, n_samples=10, n_steps=10)

if results:
	print("Generated samples,")
	for i, result in enumerate(results, 1):
		print(f"  {i}. {prompt} {result[-1].proposal.text} (n_steps={len(result)})")
else:
	print("Failed to generate any samples")

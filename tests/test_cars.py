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
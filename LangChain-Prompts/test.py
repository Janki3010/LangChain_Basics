from langchain_core.prompts import load_prompt


template = load_prompt("template.json")
print(template)
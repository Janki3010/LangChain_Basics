import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

gpt_model = os.getenv("OPENAI_MODEL")

model = ChatOpenAI(model=str(gpt_model))

prompt_template1 = PromptTemplate(
    template="Give me details of {topic}",
    input_variables=["topic"]
)

prompt_template2 = PromptTemplate(
    template="summarize in 1 line from this {text}",
    input_variables=["text"]
)
output_parser = StrOutputParser()

chain = prompt_template1 | model | output_parser | prompt_template2 | model | output_parser

result = chain.invoke({"topic": "Leo Messi"})

print(result)

chain.get_graph().print_ascii()

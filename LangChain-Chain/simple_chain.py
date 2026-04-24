from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

gpt_model = os.getenv("OPENAI_MODEL")
model = ChatOpenAI(model=str(gpt_model))

prompt = PromptTemplate(
    template="Tell me 2 lines about {topic}",
    input_variables=['topic']
)

output = StrOutputParser()

chain = prompt | model | output

result = chain.invoke({"topic": "Football"})
print(result)

chain.get_graph().print_ascii()
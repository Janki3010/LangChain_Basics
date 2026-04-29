# using RunnablePassthrough we can get same output as input

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough, RunnableSequence, RunnableParallel
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOpenAI(model=str(os.getenv("OPENAI_MODEL")))

# passthrough = RunnablePassthrough()
#
# print(passthrough.invoke({"name": "THV"}))

prompt1 = PromptTemplate(
    template="Generate a joke about {topic}",
    input_variables=["topic"]
)

prompt2 = PromptTemplate(
    template="Explain this joke {text}",
    input_variables=["text"]
)

parser = StrOutputParser()

generate_chain = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "explain": RunnableSequence(prompt2, model, parser)
})

chain = RunnableSequence(generate_chain, parallel_chain)
res = chain.invoke({"topic": "Real Madrid"})

print(res)
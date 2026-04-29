from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOpenAI(model=str(os.getenv("OPENAI_MODEL")))

prompt1 = PromptTemplate(
    template="Generate a twitter post about {topic}",
    input_variables=["topic"]
)
prompt2 = PromptTemplate(
    template="Generate a linkdin post about {topic}",
    input_variables=["topic"]
)

parser = StrOutputParser()

chain = RunnableParallel({
    'X': RunnableSequence(prompt1, model, parser),
    'linkdin': RunnableSequence(prompt2, model, parser)
})

res = chain.invoke({"topic": "AI"})

print(res)

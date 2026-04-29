# RunnableBranch work for condition , like if,elif, else

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch, RunnableSequence, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOpenAI(model=str(os.getenv("OPENAI_MODEL")))

prompt1 = PromptTemplate(
    template = "create detail report of this {topic}",
    input_variables=["topic"]
)
prompt2 = PromptTemplate(
    template = "Generate summary of this {text}",
    input_variables=["text"]
)

parser = StrOutputParser()

des_topic_chain = RunnableSequence(prompt1, model, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split()) > 500, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

chain = RunnableSequence(des_topic_chain, branch_chain)

res = chain.invoke({"topic": "Leo Messi"})

print(res)
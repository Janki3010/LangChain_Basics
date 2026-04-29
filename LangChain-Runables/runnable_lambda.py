"""using RunnableLambda use to convert normal python fun to runnable so
   it's allow you to apply custome python fun within an AI pipline"""

from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda, RunnableSequence, RunnableParallel, RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

def find_length(joke):
    return len(joke)

model = ChatOpenAI(model=str(os.getenv("OPENAI_MODEL")))

prompt = PromptTemplate(
    template="Generate a joke about {topic}",
    input_variables=["topic"],
)
parser = StrOutputParser()

generate_joke = RunnableSequence(prompt, model, parser)

parallel_chain = RunnableParallel({
    "joke": RunnablePassthrough(),
    "len": RunnableLambda(find_length)
})

chain = RunnableSequence(generate_joke, parallel_chain)

res = chain.invoke({"topic": "USA"})
final_res = f"Joke: {res.get('joke')} \n Words count: {res.get('len')}"
print(final_res)

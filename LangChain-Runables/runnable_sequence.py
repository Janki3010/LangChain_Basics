from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

model = ChatOpenAI(model=str(os.getenv("OPENAI_MODEL")))

prompt1 = PromptTemplate(
    template="Tell me joke about {topic}",
    input_variables=["topic"]
)


prompt2 = PromptTemplate(
    template="Please explain me this joke {text} with the complete joke",
    input_variables=["text"]
)

parser = StrOutputParser()

runnable_sequence = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

res = runnable_sequence.invoke({"topic": "Real Madrid"})

print(res)
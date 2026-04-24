import os

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableBranch, RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

load_dotenv()

gpt_model = os.getenv("OPENAI_MODEL")

class FeedBack(BaseModel):
    sentiment: Literal["positive", "negative"] = Field(description="Give the sentiment of the input feedback positive or negative.")

model = ChatOpenAI(model=str(gpt_model))
parser = StrOutputParser()
parser2 = PydanticOutputParser(pydantic_object=FeedBack)

prompt1 = PromptTemplate(
    template="Classify the sentiment of the input feedback text into positive or negative \n {formate_instruction}. \n {feedback}",
    input_variables=["feedback"],
    partial_variables={'formate_instruction':parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

# res = classifier_chain.invoke({"feedback": "This phone quality is Nice"})
# print(res)

prompt2 = PromptTemplate(
    template="Write an appropriate response to this positive feedback. \n {feedback}",
    input_variables=["feedback"],
)
prompt3 = PromptTemplate(
    template="Write an appropriate response to this negative feedback. \n {feedback}",
    input_variables=["feedback"],
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment == "positive", prompt2 | model | parser),
    (lambda x:x.sentiment == "negative", prompt3 | model | parser),
    RunnableLambda(lambda x: "could not find sentiment")
)

chain = classifier_chain | branch_chain

result = chain.invoke({"feedback": "This phone quality is very bad."})

print(result)

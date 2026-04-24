from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnableParallel
from langchain_openai import ChatOpenAI

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

model1 = ChatOpenAI()
model2 = ChatAnthropic(model_name='claude-sonnet-4-6')

prompt1 = PromptTemplate(
    template="Generate short & simple notes from provided text \n {text}",
    input_variables=["text"],
)
prompt2 = PromptTemplate(
    template="Generate 5 Q&A from provided text \n {text}",
    input_variables=["text"],
)
prompt3 = PromptTemplate(
    template="Combine this both notes & q&a quiz in one document \n {text} \n {quiz}",
    input_variables=["text", "quiz"],
)

parser = StrOutputParser()

parallel_chain =  RunnableParallel({
    'text': prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

simple_chain = prompt3 | model1 | parser

chain = parallel_chain | simple_chain

text = """"
With RAG, the LLM is able to leverage knowledge and information that is not necessarily in its weights by providing it access to external knowledge sources such as databases.
It leverages a retriever to find relevant contexts to condition the LLM, in this way, RAG is able to augment the knowledge-base of an LLM with relevant documents.
The retriever here could be any of the following depending on the need for semantic retrieval or not:
"""

result = chain.invoke({"text": text})
print(result)

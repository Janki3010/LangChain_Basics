from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI()

parse = JsonOutputParser()

template = PromptTemplate(
    template="Give me a name, age , book_name & country of a fictional person from ay book \n {format_instruction}",
    partial_variables={'format_instruction': parse.get_format_instructions()}
)

chain = template | model | parse

result = chain.invoke({})
print(result)
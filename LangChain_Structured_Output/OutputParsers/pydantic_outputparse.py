from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o")

class Person(BaseModel):
    name: str = Field(description="Name of person")
    age: int = Field(description="Age of person", gt=18)
    city: str = Field(description="City of person")
    country: str = Field(description="Country of person")
    book_name: str = Field(description="Name of book")

parser = PydanticOutputParser(pydantic_object=Person)

template = PromptTemplate(
    template="Generate the name, age and city of a fictional {place} person, It should come from any great book \n {format_instruction}",
    input_variables=['place'],
    partial_variables={'format_instruction':parser.get_format_instructions()},
)
print(template)
chain = template | model | parser

result = chain.invoke({'place': 'Indian'})

print(result)



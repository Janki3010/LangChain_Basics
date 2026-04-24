from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
# from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

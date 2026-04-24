from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

query = 'Delhi is capital of India'

result = embedding.embed_query(query)

print(str(result))
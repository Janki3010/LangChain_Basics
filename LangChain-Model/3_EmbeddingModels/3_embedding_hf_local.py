from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

query = "Delhi is the capital of India"

result = embedding.embed_query(query)
print(str(result))
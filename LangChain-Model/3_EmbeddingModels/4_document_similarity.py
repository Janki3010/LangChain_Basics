from langchain_openai import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=300)

documents = [
    "Lionel Messi is one of the greatest footballers of all time.",
    "Neymar Jr. is known for his incredible skills and creativity.",
    "Pedri is a talented young midfielder with excellent vision.",
    "Cristiano Ronaldo is famous for his goal-scoring and athleticism.",
    "Sunil Chhetri is one of India's greatest footballers and top scorers."
]

query = "Who is GOAT in footballers?"

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]
index, score = sorted(list(enumerate(scores)), key=lambda item: item[1])[-1]

print(query)
print(documents[index])



import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda, RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()

video_id = "GScUjc-A4yE"

api = YouTubeTranscriptApi()

try:
    transcript_list = api.fetch(video_id)
    transcript = " ".join(snippet.text for snippet in transcript_list.snippets)
    # print(transcript)
except Exception as e:
    print(f"Error: {e}")


splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.create_documents([transcript])

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vector_store = FAISS.from_documents(chunks, embeddings)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 2})

llm = ChatOpenAI(model=str(os.getenv("OPENAI_MODEL")), temperature=0.2)

prompt = PromptTemplate(
    template="""
    You are a helpful assistant.
    Answer only from the provided transcript context.
    If the context is insufficient, just say you don't know.
    
    {context}
    Question: {question}
    """
    ,
    input_variables = ['context', 'question']
)

# question = "How AI agents works? What is fundamental?"
# retrieved_docs = retriever.invoke(question)
# context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
# final_prompt = prompt.invoke({"context": context_text, "question": question})
# answer = llm.invoke(final_prompt)

# print(answer.content)

def format_context(retrieved_docs):
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text

parallel_chain = RunnableParallel({
    "context": retriever | RunnableLambda(format_context),
    "question": RunnablePassthrough()
})

str_output = StrOutputParser()

chain = parallel_chain | prompt | llm | str_output
result = chain.invoke("How AI agents works? What is fundamental?")
print(result)
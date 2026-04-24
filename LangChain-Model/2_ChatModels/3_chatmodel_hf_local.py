
# from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv

load_dotenv()

sec_key = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN")


# llm = HuggingFacePipeline.from_model_id(
#     model_id='TinyLlama/TinyLlama-1.1B-Chat-v0.6',
#     task='text-generation',
#     pipline_kwargs=dict(
#         temperature=0.5,
#         max_new_tokens=100
# )
# )

llm = HuggingFaceEndpoint(
    repo_id='meta-llama/Llama-3.1-8B-Instruct',
    task='text-generation',
    huggingfacehub_api_token=sec_key,
    provider="auto" 
)

model = ChatHuggingFace(llm=llm)
result = model.invoke("What is AI Agents?")

print(result.content)

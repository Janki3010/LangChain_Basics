from dotenv import load_dotenv
from langchain_core.prompts import load_prompt
from langchain_openai import ChatOpenAI
import streamlit as st

load_dotenv()
model = ChatOpenAI()

st.header("Research Tool")

paper_input = st.selectbox("Select Research Paper Name", ["GPT-3: Language Models", "BERT: Pre-training of Deep Bidirectional Transformers", "Different Models Beat GANs on Image Synthesis"])

style_input = st.selectbox("Select Explanation Style", ["Technical", "Mathematical"])

length_input = st.selectbox("Select Explanation Length", ["Short (1-2 paragraph)", "Medium (3-5 paragraph)", "Long (detailed explanation)"])

template = load_prompt("template.json")

# prompt = template.invoke({
#     'paper_input':paper_input,
#     'style_input':style_input,
#     'length_input':length_input,
# })
#
# if st.button("Summerize"):
#     result = model.invoke(prompt)
#     st.write(result.content)

if st.button("Summerize"):
    chain = template | model
    result = chain.invoke({
        'paper_input': paper_input,
        'style_input': style_input,
        'length_input': length_input
    })
    st.write(result.content)
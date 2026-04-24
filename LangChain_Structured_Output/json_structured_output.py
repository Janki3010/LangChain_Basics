from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI()
json_schema = {
    "title": "Review",  # REQUIRED for LangChain tool usage
    "description": "Extract structured information from a product review",
    "type": "object",
    "properties": {
        "key_themes": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Write down all the key themes discussed in the review in a list"
        },
        "summary": {
            "type": "string",
            "description": "A brief summary of the review"
        },
        "sentiment": {
            "type": "string",
            "enum": ["negative", "positive", "neutral"],
            "description": "Return sentiment of the review either negative, positive, or neutral"
        },
        "pros": {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "description": "Write down all the pros inside the review in a list"
        },
        "cons": {
            "type": ["array", "null"],
            "items": {"type": "string"},
            "description": "Write down all the cons inside the review in a list"
        },
        "name": {
            "type": ["string", "null"],
            "description": "The name of the review"
        }
    },
    "required": ["key_themes", "summary", "sentiment"],
    "additionalProperties": False
}

structured_output = model.with_structured_output(json_schema)
result = structured_output.invoke("""I recently purchased this wireless headset and overall I’m quite impressed. The sound quality is clear and immersive, especially while watching movies and listening to music. Battery life is excellent—it easily lasts a full day of use without needing a recharge. 
The build quality feels solid and premium, and the ear cushions are comfortable even during long sessions.
However, there are a few downsides. The Bluetooth connection occasionally drops when switching between devices, 
which can be frustrating. Also, the microphone quality is average and not ideal for professional calls. The price is slightly on the higher side compared to similar products.
Overall, it’s a great product for entertainment and casual use, but could use improvements in connectivity and mic performance.
""")
print(result)
from typing import TypedDict, Annotated, Optional
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

# schema
class Review(TypedDict):
    key_themes: Annotated[list[str], "Write down all the key themes discussed in the review in a list"]
    summary: Annotated[str, 'A brief summary of the review']
    sentiment: Annotated[str, 'Return sentiment of the review either negative, positive, or neutral']
    pros: Annotated[Optional[list[str]], "Write down all the pros inside the review in a list"]
    cons: Annotated[Optional[list[str]], "Write down all the cons inside the review in a list"]
    name: Annotated[Optional[str], 'The name of the review']

structured_model = model.with_structured_output(Review)

result = structured_model.invoke("The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Hoping for software update to fix this.")
print(result)
print(result.get("summary"))
print(result.get("sentiment"))
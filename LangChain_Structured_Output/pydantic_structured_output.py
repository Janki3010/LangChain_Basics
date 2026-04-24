from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Optional

load_dotenv()

class Review(BaseModel):
    key_themes: list[str] = Field(description="Write down all the key themes discussed in the review in a list")
    summary: str = Field(description="A brief summary of the review")
    sentiment: str = Field(description="Return sentiment of the review either negative, positive, or neutral")
    pros: Optional[list[str]] = Field(default=None, description="Write down all the pros inside the review in a list")
    cons: Optional[list[str]] = Field(default=None, description="Write down all the cons inside the review in a list")
    name: Optional[str] = Field(default=None, description="The name of the review")


model = ChatOpenAI()
structured_model = model.with_structured_output(Review)
result = structured_model.invoke("The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Hoping for software update to fix this. Reviewed By Janki Patel")

print(result)
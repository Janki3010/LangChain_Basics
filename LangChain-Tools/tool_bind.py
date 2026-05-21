import json
import os

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, InjectedToolArg
from langchain_core.messages import HumanMessage
from typing import Annotated
import requests
from dotenv import load_dotenv

load_dotenv()
# tool create

@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """
    This function fetches the currency conversion factor between a given base currency and a target currency.
    """
    url = f'https://v6.exchangerate-api.com/v6/{os.getenv("EXCHANGE_API_KEY")}/pair/{base_currency}/{target_currency}'
    response = requests.get(url)
    return response.json()

# print(get_conversion_factor.invoke({'base_currency': 'USD', 'target_currency': 'INR'}))

@tool
def convert(base_currency: int, conversion_rate: Annotated[float, InjectedToolArg]) -> float:
    """
    given a currency conversion rate this function calculates the target currency value from a given base currency value
    """
    return base_currency * conversion_rate

# print(convert.invoke({'base_currency': 10, 'conversion_rate': 96.8244}))
messages = []
llm  = ChatOpenAI()

llm_with_tools = llm.bind_tools([get_conversion_factor, convert])

message = [HumanMessage('What is the convertion factor between USD and INR, and based on that can you convert 45 usd to inr')]

ai_message = llm_with_tools.invoke(message)
messages.append(ai_message)

for tool_call in ai_message.tool_calls:
    # execute the 1st tool and get the value of convertion rate
    # execute the 2nd tool using the conversion rate from tool 1

    if tool_call['name'] == 'get_conversion_factor':
        tool_message1 = get_conversion_factor.invoke(tool_call)
        print(tool_message1)

        #fetch this conversion rate
        conversion_rate = json.loads(tool_message1.content)['conversion_rate']
        print(f'conversion_rate : {conversion_rate}')
        messages.append(tool_message1)

    if tool_call['name'] == 'convert':
        tool_call['args']['conversion_rate'] = conversion_rate
        tool_message2 = convert.invoke(tool_call)
        messages.append(tool_message2)

print(llm_with_tools.invoke(messages).content)

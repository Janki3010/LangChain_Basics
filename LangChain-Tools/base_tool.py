from typing import Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

class MultiplyInput(BaseModel):
    a: int = Field(description="The first number")
    b: int = Field(description="The second number")

class MultiplyTool(BaseTool):
    name: str = "Multiply"
    description: str = "Multiply two numbers"

    args_schema: Type[BaseModel] = MultiplyInput
    
    def _run(self, a: int, b: int) -> int:
        return a * b

multiply_tool = MultiplyTool()

result = multiply_tool.invoke({'a':10, 'b':2})

print(result)
print(multiply_tool.name)
print(multiply_tool.description)
print(multiply_tool.args)
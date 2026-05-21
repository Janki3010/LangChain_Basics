from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Adds two numbers"""
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Subtracts two numbers"""
    return a - b

class MathToolkit:
    def get_tools(self):
        return [add, subtract]

toolkit = MathToolkit()
tools = toolkit.get_tools()

for tool in tools:
    print(tool.name, "->", tool.description)



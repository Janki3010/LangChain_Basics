from langchain_community.tools import DuckDuckGoSearchRun, ShellTool

#Built-in tools

search_tool = DuckDuckGoSearchRun()
result = search_tool.invoke('IPL latest news , which teams qualified')
print(result)
print(search_tool.name)
print(search_tool.description)
print(search_tool.args)

shell_tool = ShellTool()
res = shell_tool.invoke('ls')
print(res)
from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from rich import print

# Tool
@tool
def get_text_length(text: str) -> int:
    """Returns the number of characters in a given text"""
    return len(text)

tools = {
    "get_text_length": get_text_length
}

# LLM
llm = ChatMistralAI(model="mistral-small-2506")

# Bind tool
llm_with_tool = llm.bind_tools([get_text_length])

messages = []

prompt = input("You: ")

messages.append(HumanMessage(content=prompt))

# First LLM call
result = llm_with_tool.invoke(messages)

messages.append(result)

# Tool execution
if result.tool_calls:

    tool_call = result.tool_calls[0]

    tool_name = tool_call["name"]

    tool_result = tools[tool_name].invoke(tool_call["args"])

    # VERY IMPORTANT
    messages.append(
        ToolMessage(
            content=str(tool_result),
            tool_call_id=tool_call["id"]
        )
    )

    # Final LLM response
    final_result = llm_with_tool.invoke(messages)

    print(final_result.content)

else:
    print(result.content)
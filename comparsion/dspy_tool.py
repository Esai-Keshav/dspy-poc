import dspy


def add(a: int, b: int) -> int:
    """Add 2 numbers"""
    return a + b


tools = [
    {
        "type": "function",
        "name": "add",
        "description": "Add 2 numbers",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {
                    "type": "int",
                    "description": "number one",
                },
                "b": {
                    "type": "int",
                    "description": "Number two",
                },
            },
            "required": ["a", "b"],
        },
    },
]

# model_2 = "smollm2:360m-instruct-q4_K_M"
# model_2 = "llama3.2:1b-instruct-q4_K_M"

lm = dspy.LM(model=f"groq/llama-3.1-8b-instant", temperature=0.5, max_tokens=300)
# lm = dspy.LM(model=f"ollama_chat/{model_2}", temperature=0.5, max_tokens=300)

dspy.configure(lm=lm)


class ToolSignature(dspy.Signature):
    """Signature for manual tool handling."""

    question: str = dspy.InputField()
    tools: list[dspy.Tool] = dspy.InputField()
    outputs: dspy.ToolCalls = dspy.OutputField()


# res = model(question="Captial of India", tools=tools)
# res = model(question="Add 5 and 2", tools=tools)
# # for call in res:
# #     print(call)

# print(res)

# print(dspy.Tool(add))

tools = {
    "add": dspy.Tool(add),
    # "calculator": dspy.Tool(calculator)
}

print(list(tools.values()))
model = dspy.ReAct(ToolSignature, tools=list(tools.values()))

response = model(question="Add 2 and 4")

# Execute the tool calls
for call in response.outputs.tool_calls:
    # Execute the tool call
    # result = call.execute()
    # For versions earlier than 3.0.4b2, use: result = tools[call.name](**call.args)
    print(f"Tool: {call.name}")
    print(f"Args: {call.args}")
    # print(f"Result: {result}")

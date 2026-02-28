import asyncio
import dspy
from dotenv import load_dotenv

load_dotenv()
lm = dspy.LM(model="groq/llama-3.1-8b-instant", max_tokens=300, temperature=0.3)
dspy.configure(lm=lm)


class ToolSignature(dspy.Signature):
    """Signature for manual tool handling."""

    question: str = dspy.InputField()
    tools: list[dspy.Tool] | None = dspy.InputField()
    outputs: dspy.ToolCalls | None = dspy.OutputField(desc="Tool calls if needed")
    answer: str | None = dspy.OutputField(
        desc="Normal response if no tool is needed, Parse the answer"
    )


predictor = dspy.streamify(dspy.Predict(ToolSignature))


tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two number",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {
                        "type": "int",
                        "description": "Number to add",
                    },
                    "y": {
                        "type": "int",
                        "description": "Number to add",
                    },
                },
                "required": ["a", "b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_therapists_by_service",
            "description": "List the therapists by their services",
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {
                        "type": "string",
                        "description": "name of the service",
                    },
                },
                "required": ["service"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_therapists",
            "description": "List all the therapists",
            "parameters": {},
        },
    },
]


async def use_streaming():
    output = predictor(question="show all therapist", tools=tools)
    return_value = None
    async for value in output:
        if isinstance(value, dspy.Prediction):
            return_value = value
        else:
            print(value)
    return return_value


output = asyncio.run(use_streaming())
print(output)

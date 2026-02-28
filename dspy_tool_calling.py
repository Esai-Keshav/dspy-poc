import dspy
from dotenv import load_dotenv
import orjson
from rich import print
from rich.console import Console
from rich.markdown import Markdown
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")

# Specify the experiment you just created for your GenAI application.
mlflow.set_experiment("TEST")


load_dotenv()
mlflow.dspy.autolog()
lm = dspy.LM(
    model="groq/llama-3.1-8b-instant",
    max_tokens=300,
    temperature=0.3,
)
# lm = dspy.LM(model="ollama_chat/llama3.2:1b-instruct-q4_K_M", max_tokens=500, temperature=0.3)
dspy.configure(lm=lm)


class ToolSignature(dspy.Signature):
    """Signature for manual tool handling."""

    question: str = dspy.InputField()
    tools: list[dspy.Tool] | None = dspy.InputField()
    outputs: dspy.ToolCalls | None = dspy.OutputField(desc="Tool calls if needed")
    answer: str | None = dspy.OutputField(
        desc="Normal response if no tool is needed, Parse the answer. "
    )


class FinalAnswerSignature(dspy.Signature):
    """Signature for for response."""

    question: str = dspy.InputField()
    tool_response: dict | None = dspy.InputField(desc="Result from tool calling")
    answer: str = dspy.OutputField(
        desc="Parse answer in readable form and present it in neat format and make it easy to read and use markdown"
    )


tools = [
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
                        "description": "Name of the service",
                    },
                },
                "required": ["service"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_therapist",
            "description": "List all the therapists",
            "parameters": {},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_services",
            "description": "List all the Services",
            "parameters": {},
        },
    },
]


def list_therapists_by_service(service: str):
    if service in ["burnout", "stress"]:
        return orjson.dumps(
            {"name": "Esai", "exp": 2, "service": ["stress", "burnout"]}
        )

    return orjson.dumps(
        {
            "name": "Ram",
            "exp": 4,
            "services": ["stress", "anger issues"],
        }
    )


def list_services():
    return orjson.dumps(["stress", "burnout", "anger issues"])


def list_therapists():
    return orjson.dumps(
        [
            {"name": "Ram", "exp": 4, "services": ["stress", "anger issues"]},
            {"name": "Esai", "exp": 2, "services": ["stress", "burnout"]},
        ]
    )


predictor = dspy.Predict(ToolSignature)
final_ans = dspy.Predict(FinalAnswerSignature)
# final_ans = dspy.ChainOfThought(FinalAnswerSignature)

# question = "what is langchain "
# question = "i want therapist for stress"
question = "list therapists"
# question = "list all  the services available "
# question="add 2 and 5",
# print(response)
# Execute the tool calls

response = predictor(
    question=question,
    tools=tools,
)

# response
print(question)

tool_calls = getattr(response.outputs, "tool_calls", None)

if tool_calls:
    # print('hi')
    ai = None

    for call in response.outputs.tool_calls:
        print(f"Tool: {call.name}")
        print(f"Args: {call.args}")

        if call.name == "list_therapists_by_service":
            result = list_therapists_by_service(**call.args)
            print(f"Result : {result}")
            ai = final_ans(question=question, tool_response={"result": result})

        if call.name == "list_therapists":
            result = list_therapists(**call.args)
            print(f"Result : {result}")
            ai = final_ans(question=question, tool_response={"result": result})
        if call.name == "list_services":
            result = list_services(**call.args)
            print(f"Result : {result}")
            ai = final_ans(question=question, tool_response={"result": result})
    print(ai)


else:
    print(response.answer)

cli = Console()
cli.print(Markdown(ai.answer))

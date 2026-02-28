import asyncio
import dspy

lm = dspy.LM(model="ollama_chat/llama3.2:1b-instruct-q4_K_M")

dspy.settings.configure(lm=lm)


predict = dspy.Predict("question -> answer")

# Enable streaming for the 'answer' field
stream_predict = dspy.streamify(
    predict,
    stream_listeners=[dspy.streaming.StreamListener(signature_field_name="answer")],
)


async def read_output_stream():
    output_stream = stream_predict(question="Why did use linux vs windows?")

    async for chunk in output_stream:
        print(chunk)
    print("last")


asyncio.run(read_output_stream())

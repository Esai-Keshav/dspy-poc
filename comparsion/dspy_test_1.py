import time

start = time.perf_counter()


import dspy

model_2 = "smollm2:360m-instruct-q4_K_M"

lm = dspy.LM(
    model=f"ollama_chat/{model_2}",
    temperature=0.5,
)
# dspy.configure(lm=lm)

dspy.settings.configure(lm=lm)


class QA(dspy.Signature):
    """Answer questions."""

    question = dspy.InputField()
    answer = dspy.OutputField()


class BasicQA(dspy.Module):
    def __init__(self):
        super().__init__()
        self.qa = dspy.ChainOfThought(QA)

    def forward(self, question):
        return self.qa(question=question)


qa = BasicQA()
result = qa("What is the capital of France?")
# print(result.answer)
end = time.perf_counter()
print(f"Time taken: {end - start:.4f} seconds")

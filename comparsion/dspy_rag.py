from time import perf_counter

import dspy
import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer


class RAG(dspy.Signature):
    """You are Teacher help students with their doubts
    Use context to answer the question
    Be concise
    """

    context: list = dspy.InputField(desc="Context")
    question: str = dspy.InputField(desc=" question")

    answer: str = dspy.OutputField(
        desc="Provide Structured the answer and present it clearly and explain it with a story"
    )


class RAGPipeline(dspy.Module):
    def __init__(self):
        load_dotenv()
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        # PostgreSQL connection
        self.conn = psycopg.connect("postgresql://esai:1234@localhost:5432/postgres")
        register_vector(self.conn)
        llm = dspy.LM(
            model="groq/llama-3.1-8b-instant", max_tokens=400, temperature=0.3
        )
        # llm = dspy.LM(model="ollama_chat/smollm2:360m-instruct-q4_K_M")
        self.cot = dspy.Predict(RAG)
        dspy.configure(lm=llm)

    def retrieve(self, query, k=3):
        qvec = self.embedder.encode(query).tolist()
        # print(qvec)

        sql = """
        SELECT document
        FROM langchain_pg_embedding
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """
        # print(type(qvec), len(qvec))
        rows = self.conn.execute(sql, (qvec, k)).fetchall()
        # print(rows)

        return "\n".join(r[0] for r in rows)

    def forward(self, question):
        context = self.retrieve(query=question)
        res = self.cot(context=context, question=question)

        return res


if __name__ == "__main__":
    start = perf_counter()
    rag = RAGPipeline()
    end = perf_counter()
    print(f"Time taken to load packages: {end - start:.4f} seconds")

    start = perf_counter()
    answer = rag("explain Content-based recommendation for exam")
    print(answer)
    end = perf_counter()
    print(f"Time taken: {end - start:.4f} seconds")

    start = perf_counter()
    answer = rag("explain context-based recommendation for exam")
    print(answer)
    end = perf_counter()
    print(f"Time taken: {end - start:.4f} seconds")

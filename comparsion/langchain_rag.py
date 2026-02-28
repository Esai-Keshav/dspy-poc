from time import perf_counter

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres import PGVector


class RAGPipeline:
    def __init__(self):
        load_dotenv()
        # self.llm = OllamaLLM(model="smollm2:360m-instruct-q4_K_M")
        self.llm = ChatGroq(
            model="llama-3.1-8b-instant", max_tokens=300, temperature=0.3
        )

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        connection = "postgresql+psycopg2://esai:1234@localhost:5432/postgres"

        collection_name = "book_rag"

        self.vector_store = PGVector(
            embeddings=embeddings,
            collection_name=collection_name,
            connection=connection,
            use_jsonb=True,
        )
        # logger.info("config Loaded")/

    def retrieve(self, query):
        docs = self.vector_store.similarity_search(query, k=3)

        return [doc.page_content for doc in docs]

    def run(self, question):
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """

            ## Role

            - You are Teacher help students with their doubts.
            - Be Concise
            - Do not guess and  assume
            - Do not hallucinate

            ## Answer only based on this only:
            {context}

            """,
                ),
                ("human", "{question}"),
            ]
        )

        model = prompt | self.llm
        res = model.invoke({"context": self.retrieve(question), "question": question})
        # logger.info("response ok")
        return res


if __name__ == "__main__":
    # logger.debug("START")
    start = perf_counter()
    rag = RAGPipeline()
    end = perf_counter()
    print(f"Time taken to load packages: {end - start:.4f} seconds")

    start = perf_counter()
    # with Profile as pr:
    print(rag.run("What is context aware recommedation").content)

    end = perf_counter()
    print(f"Time taken: {end - start:.4f} seconds")
    # logger.debug("STOP")

    start = perf_counter()
    print(rag.run("Explain What is hyrid recommedation system").content)
    end = perf_counter()
    print(f"Time taken: {end - start:.4f} seconds")

    # logger.debug("STOP")

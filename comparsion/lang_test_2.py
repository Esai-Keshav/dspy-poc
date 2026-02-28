import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.embeddings import Model2vecEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM

embedding = Model2vecEmbeddings("minishlab/potion-base-8M")

model = OllamaLLM(model="smollm2:360m-instruct-q4_K_M")


def vector_store(texts):
    dim = len(embedding.embed_query("hello world"))

    index = faiss.IndexFlatIP(dim)
    db = FAISS(
        embedding_function=embedding,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )

    db.add_texts([texts])
    return True


def generate_reply(context, query): ...


if __name__ == "__main__":
    pass

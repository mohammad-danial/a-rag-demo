import os
import shutil

from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from loguru import logger

CHROMA_PATH = "chroma"
DATA_PATH = "./generated_stories"


def initialize_data_store():
    documents = load_text_documents()
    chunks = chunk_documents(documents)
    store_chunks_in_chroma(chunks)


def load_text_documents():
    loader = DirectoryLoader(DATA_PATH, glob="*.md")
    documents = loader.load()
    return documents


def chunk_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    return chunks


def store_chunks_in_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embedding_model = "llama3"
    embedding = OllamaEmbeddings(model=embedding_model)
    db = Chroma.from_documents(
        chunks,
        embedding,
        persist_directory=CHROMA_PATH
    )
    db.persist()
    logger.info(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


def get_answer(query_text):
    prompt_template_text = """
    Act as a skilled and knowledgeable expert in the field of literature.
    Use this context:
    {context}

    To be able to answer this question in less than 100 words: {question}
    """

    embedding_model = "llama3"
    embedding_function = OllamaEmbeddings(model=embedding_model)
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embedding_function
    )

    results = db.similarity_search_with_relevance_scores(query_text, k=3)

    context_text = "\n\n---\n\n".join(
        [doc.page_content for doc, _score in results]
    )
    prompt_template = ChatPromptTemplate.from_template(prompt_template_text)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = OllamaLLM(model="llama3")
    response = model.invoke(prompt)

    found_sources = [
        {
            "Source": document.metadata.get("source", "Unknown"),
            "Score": _score,
        }
        for document, _score in results
    ]

    logger.info(f"Found sources:\n{found_sources}")
    return response


def interactive_chat():
    print("Ask questions about the stories. Type 'exit' to quit.")
    while True:
        try:
            question = input(">> You: ")
            if question.lower() == "exit":
                break
            formatted_response = get_answer(question)
            print(f">> Assistant: {formatted_response}")
        except KeyboardInterrupt:
            print("\nExiting...")
            break


if __name__ == "__main__":
    initialize_data_store()
    interactive_chat()

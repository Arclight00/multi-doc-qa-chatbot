import itertools
import os

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import CTransformers
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    DirectoryLoader,
)


def load_document_dir(directory):
    loader = DirectoryLoader(directory)
    docs = loader.load()
    return docs


def load_document(file_path):
    name, extension = os.path.splitext(file_path)

    if extension == ".pdf":
        print(f"Loading {file_path}")
        loader = PyPDFLoader(file_path)
    elif extension == ".docx":
        print(f"Loading {file_path}")
        loader = Docx2txtLoader(file_path)
    else:
        print("Document format is not supported!")
        return None

    docs = loader.load()
    return docs


def split_docs(docs, chunk_size=1024, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
    )
    return text_splitter.split_documents(docs)


def chunk_iterable(iterable, batch_size=100):
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))


def prepare_vectors_and_metadata(chunk, embed_model):
    vectors_and_metadata = []
    for i, doc in enumerate(chunk):
        doc_id = f"{doc.metadata['source']}-{i}"
        vector = embed_model.encode([doc.page_content])[0]
        metadata = {"text": doc.page_content, "source": doc.metadata["source"]}
        vectors_and_metadata.append((doc_id, vector, metadata))
    return vectors_and_metadata


def get_embeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu"):
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device},
    )
    embed_dim = embeddings.client[1].word_embedding_dimension
    return embeddings, embed_dim


def load_model(
        model_path="model/llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.7,
):
    """
    Load a locally downloaded model.

    Parameters:
        model_path (str): The path to the model to be loaded.
        model_type (str): The type of the model.
        max_new_tokens (int): The maximum number of new tokens for the model.
        temperature (float): The temperature parameter for the model.

    Returns:
        CTransformers: The loaded model.

    Raises:
        FileNotFoundError: If the model file does not exist.
        SomeOtherException: If the model file is corrupt.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model file found at {model_path}")

    llm = CTransformers(
        model=model_path,
        model_type=model_type,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    return llm


def get_llama_2_prompt(messages) -> str:
    """Generate Llama 2 prompt for the given messages."""
    context = ""
    question = ""

    for message in messages:
        role = message.role
        content = message.content or ""

        if role.lower() == "system":
            context = content.strip()
        elif role.lower() == "user":
            question = content.strip()

    prompt = f"<s> [INST] <<SYS>> Use the following pieces of context to answer the user's question. " \
             f"If the answer is not contained within the context below, just say 'I don't know the answer', " \
             f"don't try to make up an answer.\n\nContext: {context}\nQuestion: {question}" \
             f"\n\nOnly return the most relevant answer below if found in the context and nothing else" \
             f"\nAnswer:</s>"

    return prompt

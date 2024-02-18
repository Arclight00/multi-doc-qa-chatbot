from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.llms import CTransformers
from langchain_openai import ChatOpenAI

from config import *
from db_connection.pinecone_db import PineconeDB
from utils import (
    load_document_dir,
    load_document,
    get_embeddings,
    split_docs,
)


class DataIngestion:

    def __init__(self, dir_path: str = "", file_path: str = ""):
        self.dir_path = dir_path
        self.file_path = file_path

    def read_documents(self):
        try:
            if self.dir_path and self.file_path:
                raise ValueError(
                    "Only one of dir_path or file_path should be provided, not both."
                )

            if self.dir_path:
                docs = load_document_dir(self.dir_path)
            elif self.file_path:
                docs = load_document(self.file_path)
            else:
                raise ValueError("Either dir_path or file_path must be provided.")

            print(f"Successfully loaded {len(docs)} documents.")
            return docs
        except Exception as e:
            print(f"Error during document loading: {str(e)}")
            return None

    def split_embed_store(self, docs):
        try:
            chunked_data = split_docs(docs, chunk_size=1024, chunk_overlap=50)
            embed_model, dim = get_embeddings()
            pc = PineconeDB(pinecone_index_name, dim)
            pc.insert_embeddings(chunked_data, embed_model)
            print("Data embedded and stored successfully.")
        except Exception as e:
            print(f"Error during data processing and storage: {str(e)}")

    @staticmethod
    def delete_knowledgebase(index_name=pinecone_index_name):
        try:
            pc = PineconeDB(index_name)
            pc.delete_pinecone_index(index_name)
        except Exception as e:
            print(f"Error while deleting the index, Error: {e}")

    def run(self):
        docs = self.read_documents()
        if docs:
            self.split_embed_store(docs)


class QAChatbot:
    gpt_prompt_template = """Use the following pieces of context to answer the users question.
    If the answer is not contained within the context below, say "I don't know the answer", 
    don't try to make up an answer. 

    Context: {context}
    Question: {question}
    
    Answer: 
    """

    llama_prompt_template = """
    [INST]<<SYS>>
    You are an expert Q&A system that is trusted around the world. 
    Always answer the question from the context provided below, and not prior knowledge.
    Some rules to follow:
    1. Never directly reference the given context in your answer.
    2. Avoid statements like 'Based on the context, ...' or 'According to the given context, ...' 
    or anything along those lines.
    3. If the answer is not found in the provided context, say "I don't know the answer".
    4. Be precise in your answer, no need to explain the answer until explicitly asked in the question.  
    <</SYS>>
    
    Context: {context}
    Question: {question}

    Answer:[/INST]
    """

    def __init__(self, model_type="", top_k=3):
        if model_type == "" or model_type not in models_used:
            raise ValueError(
                "Invalid model_type. Supported types are 'gpt' and 'llama'."
            )
        self.model_type = model_type
        self.top_k = top_k

    def load_model(
        self,
        max_new_tokens=800,
        temperature=0,
    ):
        if self.model_type == "llama-2-7b-chat.ggmlv3.q4_1.bin":
            model_path = "llms/llama-2-7b-chat.ggmlv3.q4_1.bin"
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"No model file found at {model_path}")

            print(os.path.abspath(model_path))
            llm = CTransformers(
                model=os.path.abspath(model_path),
                model_type="llama",
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                config={"context_length": 2048},
            )

        else:
            llm = ChatOpenAI(
                model="gpt-3.5-turbo",
                max_tokens=max_new_tokens,
                temperature=temperature,
            )

        return llm

    def create_retrieval_qa_chain(self, llm, prompt, db):

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=db.as_retriever(search_kwargs={"k": self.top_k}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt},
        )
        return qa_chain

    def create_custom_prompt(self):
        if self.model_type == "llama":
            prompt = PromptTemplate(
                template=self.llama_prompt_template,
                input_variables=["context", "question"],
            )
        else:
            prompt = PromptTemplate(
                template=self.gpt_prompt_template,
                input_variables=["context", "question"],
            )
        return prompt

    def create_retrieval_qa_bot(
        self,
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
        device="cpu",
    ):
        try:
            embeddings, _ = get_embeddings(embedding_model_name, device)
        except Exception as e:
            raise Exception(
                f"Failed to load embeddings with model name {embedding_model_name}: {str(e)}"
            )

        pc = PineconeDB(pinecone_index_name)
        vector_store = pc.fetch_embeddings(embeddings)
        try:
            llm = self.load_model()
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")

        qa_prompt = self.create_custom_prompt()

        try:
            qa = self.create_retrieval_qa_chain(
                llm=llm, prompt=qa_prompt, db=vector_store
            )
        except Exception as e:
            raise Exception(f"Failed to create retrieval QA chain: {str(e)}")
        return qa

    def chat(self, query):
        try:
            qa_bot_instance = self.create_retrieval_qa_bot()
            bot_response = qa_bot_instance({"query": query})
            print("{} Top 3 results {}".format("*" * 100, "*" * 100))
            for i in bot_response["source_documents"]:
                print("*" * 200)
                print(i.page_content)
                print("*" * 200)
            print("Final Answer: ", bot_response["result"])
            return bot_response
        except Exception as e:
            print(f"Error while fetching response from chatbot, Error: {e}")
            raise

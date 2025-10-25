import os
import dotenv
from langchain.chat_models.base import BaseChatModel
from langchain.chat_models import init_chat_model

dotenv.load_dotenv()


def get_input(text: str) -> str:
    return input(text)


def init_llm(model: str = "gpt-4o-550k-2", temperature: float = 0.0) -> BaseChatModel:
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    llm = init_chat_model(model=model, temperature=temperature)
    return llm

def init_vector_store(collection_name: str='example_collection', persist_directory: str='./chroma_langchain_db') -> object:
    if os.getenv('ENV') != 'local':
        raise ValueError("Vector store initialization is only supported in local environment.")
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-large"),
        persist_directory=persist_directory,
    )
    return vector_store
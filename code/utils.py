import base64
import io
import os

import dotenv
from langchain.chat_models import init_chat_model
from langchain.chat_models.base import BaseChatModel
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from PIL import Image
from tools import execute_python_code

dotenv.load_dotenv()


def image_to_base64(image: Image) -> str:
    """Decode Image to base64 string through bytes buffer"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    im_bytes = buffered.getvalue()
    return base64.b64encode(im_bytes)


def init_chat_llm(model: str = "gpt-4o-550k-2", temperature: float = 0.0, use_tools: bool = True) -> BaseChatModel:
    """Initialize a chat-based LLM.

    Args:
        model (str, optional): The model to use. Defaults to "gpt-4o-550k-2".
        temperature (float, optional): The temperature to use for sampling. Defaults to 0.0.
        use_tools (bool, optional): Whether to use tools. Defaults to True.

    Raises:
        ValueError: If the OPENAI_API_KEY environment variable is not set.

    Returns:
        BaseChatModel: The initialized chat model.
    """
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable not set.")
    llm = init_chat_model(model=model, temperature=temperature)
    if use_tools:
        llm = llm.bind_tools([execute_python_code])
    return llm


def init_vector_store(
    collection_name: str = "example_collection",
    persist_directory: str = "./chroma_langchain_db",
    embedding_model: str = "text-embedding-3-large",
) -> object:
    """Initialize a vector store.

    Args:
        collection_name (str, optional): The name of the collection. Defaults to "example_collection".
        persist_directory (str, optional): The directory to persist the vector store. Defaults to "./chroma_langchain_db".
        embedding_model (str, optional): The embedding model to use. Defaults to "text-embedding-3-large".
    Returns:
        object: The initialized vector store.
    """

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=OpenAIEmbeddings(model=embedding_model),
        persist_directory=persist_directory,
    )
    return vector_store

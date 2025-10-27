"""Prompts used throughout the application."""

from langchain_core.prompts import PromptTemplate

SYSTEM_PROMPT = """You are a helpful assistant tasked with analyzing some documents.
You will be given a question/inquiry related to some documents and the relevant documents.
Base your answer only on the context given, do not use outside information.
Always answer in the language of the user"""

USER_PROMPT = PromptTemplate.from_template("""User Query:
{query}
***************************
Context:
{context}
""")

SUMMARIZE_PROMPT = """Summarize the conversation thus far
"""

EXTRACT_PROMPT = """Extract the relevant fields from this CV"""

IMAGE_DESCRIPTION_PROMPT = """Describe the content of this image."""
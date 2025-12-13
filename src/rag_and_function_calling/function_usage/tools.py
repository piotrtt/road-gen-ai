# -*- coding: utf-8 -*-
"""
In this module, you should implement the actual functions to be called when the LLM chooses to use a function.
Hint:
You can find a function in `utils/semantic_search.py` that retrieves relevant documents for a given query.
"""
from ..utils.semantic_search import retrieve_relevant_documents  # noqa: F401


def get_chunks_from_docs(question: str, n_documents: int = 10) -> list:
    """
    This function takes a string as input and returns a list of the n most relevant Document objects.
    :param question: the main topic in the user's question as a string
    :param n_documents: the number of chunks to return as an integer
    :return: a list of Document objects
    """
    result = retrieve_relevant_documents(question, n_documents)
    return [item.payload for item in result]

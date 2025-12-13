"""

"""
import json
from string import Template

from ..utils.semantic_search import retrieve_relevant_documents  # noqa: F401


PROMPT_TEMPLATE = Template(
    """Context information is below.
---------------------
$context_str
---------------------
Given the context information and not prior knowledge,
answer the query precisely and concisely referring to the provided documents.
Please provide your answer markdown format.

Query: $user_input
Answer:"""
)


def build_prompt(input_str: str) -> str:
    """
    This function executes the RAG (Retrieval-Augmented Generation) workflow. It retrieves relevant code chunks for a
    given query and combines them with the query and specific instructions for the LLM.
    Ensure the retrieved outputs are formatted appropriately for the LLM.

    :param input_str: A question or task provided by the user.
    :return: A formatted prompt string for the LLM.
    """
    documents = retrieve_relevant_documents(input_str, 3)
    context_str = "\n".join([json.dumps(item.payload, indent=4) for item in documents])
    return PROMPT_TEMPLATE.substitute(context_str=context_str, user_input=input_str)

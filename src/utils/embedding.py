"""
DEPRECATED: This module is no longer used for road network similarity.

The embedding-based approach has been replaced with a custom similarity
metric (src/metrics/similarity.py) as recommended by Professor Bade.

This module is kept for reference and potential future use in other
contexts (e.g., RAG for documentation search).

See: docs/phase1_implementation_summary.md for rationale.
"""
import warnings
from openai import OpenAI
import dotenv

from sklearn.metrics.pairwise import cosine_similarity

# Deprecation warning
warnings.warn(
    "embedding.py is deprecated for road network similarity. "
    "Use src/metrics/similarity.py instead.",
    DeprecationWarning,
    stacklevel=2
)


def embed_text(text: str) -> list:
    """
    This converts a single chunk of text into an embedding.
    :param text: A chunk as a string.
    :return: a list of floats representing the embedding.
    """
    client = OpenAI(
  api_key=dotenv.get_key(dotenv.find_dotenv(), "OPENAI_API_KEY")
    )
    response = client.embeddings.create(input=[text], model="text-embedding-3-large")
    return response.data[0].embedding


def calculate_similarity(embedding1: list, embedding2: list) -> float:
    """
    This function calculates the similarity between two embeddings.
    :param embedding1: a vector.
    :param embedding2: another vector.
    :return: the similarity between the two vectors calculated with a distance metric of your choice.
    """
    return cosine_similarity([embedding1], [embedding2])[0][0]

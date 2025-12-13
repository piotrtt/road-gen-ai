"""
This module provides a common embedding client instance to use in the
semantic search exercises.
"""
from openai import OpenAI
import dotenv

from sklearn.metrics.pairwise import cosine_similarity


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

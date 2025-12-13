
"""
Module for managing Qdrant vector database collections.

This module provides utility functions to facilitate interactions with the Qdrant 
vector database, including collection lifecycle management and document ingestion. 
It ensures collections are configured with the required vector parameters 
(3072 dimensions, Cosine distance) and handles the conversion of preprocessed 
Document objects into searchable vector points via embedding generation.

"""
import uuid

from qdrant_client import models
from typing import Any
from embedding import embed_text
from qdrant import qdrant_client  # noqa: F401
from qdrant_client.models import PointStruct


def get_or_create_qdrant_collection(collection_name: str) -> str:
    """
    This function creates a new collection in the qdrant instance if it does not exist yet.
    Else it returns the collection name.
    :param collection_name as a string
    :return: the collection name as a string
    """
    if not qdrant_client.collection_exists(collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=3072, distance=models.Distance.COSINE
            ),
        )
    return collection_name


def add_documents_to_collection(collection_name: str, documents: list[Any]):
    """
    This function adds documents to a given collection in the qdrant instance.
    :param collection_name: the name of the collection
    :param documents: a list of chunked documents
    :return: None
    """
    embeddings = embed_texts([doc.chunk for doc in documents])
    points = [
        PointStruct(
            id=uuid.uuid4().hex,
            vector=embedding,
            payload={"text": text},
        )
        for embedding, text in list(zip(embeddings, documents))
    ]
    qdrant_client.upsert(collection_name, points)

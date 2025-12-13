# -*- coding: utf-8 -*-
from openai import AzureOpenAI
from qdrant_client import QdrantClient
from qdrant import qdrant_client
from OpenAI import client



def retrieve_relevant_documents(query: str, n_documents: int = 10) -> list:
    """
    This function retrieves the n most relevant documents from the Qdrant collection.
    :param query: The query to search for.
    :param n_documents: The number of documents to retrieve.
    :return: A list of documents.
    """
    oai_response = client.embeddings.create(
        input=[query], model="text-embedding-3-large"
    )
    query_embedding = oai_response.data[0].embedding
    return qdrant_client.search(
        collection_name="scenario-generation-samples", query_vector=query_embedding, limit=n_documents
    )

if __name__ == "__main__":
    docs = retrieve_relevant_documents("Roundabout", 1)
    for doc in docs:
        print(doc.payload)
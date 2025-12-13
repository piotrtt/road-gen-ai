"""

"""
from embedding import embed_text
from typing import Any
from qdrant import client  # noqa: F401


def retrieve_relevant_documents(
    collection_name: str, query: str, k: int
) -> list[Any]:
    """
    This function retrieves the most similar documents to a given query from a given collection.
    :param collection_name: the name of the collection
    :param query: the query
    :param k: the number of most similar documents to retrieve
    :return: a list of document ids
    """
    query_embedding = embed_text(query)
    print("Connecting to Qdrant...")
    return client.query_points(
        collection_name=collection_name, query=query_embedding, limit=k
    )

if __name__ == "__main__":
    docs = retrieve_relevant_documents("scenario-generation-samples", "Roundabout", 1)
    print(docs.points[0].payload["text"]["chunk"])

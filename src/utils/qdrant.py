"""
This module provides a common qdrant client instance.
"""
from qdrant_client import QdrantClient
import os

# hint: adapt paths to your needs
# If you've used the provided docker compose file adapt the path like this: QdrantClient("localhost", port=6333)
client = QdrantClient(
    url="https://59b9c848-48ce-49df-b3c2-7d9c3c882bb9.eu-central-1-0.aws.cloud.qdrant.io:6333", 
    api_key=os.getenv("QDRANT_API_KEY"),
)

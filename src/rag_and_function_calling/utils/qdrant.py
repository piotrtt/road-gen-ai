# -*- coding: utf-8 -*-
"""
This module provides a common qdrant client instance to use in the
different exercises.
"""
from qdrant_client import QdrantClient

# hint: adapt paths to your needs
# If you've used the provided docker compose file adapt the path like this: QdrantClient("localhost", port=6333)
qdrant_client = QdrantClient(
    url="https://59b9c848-48ce-49df-b3c2-7d9c3c882bb9.eu-central-1-0.aws.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.VZp3LADBpsUEOs9AzeaqzYSRF6MiZps7DBeIGWJQCyA",
)

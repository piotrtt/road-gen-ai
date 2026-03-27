"""
DEPRECATED: This module is no longer used for road network similarity.

The vector database approach has been replaced with filesystem-based
storage (src/generators/network_storage.py) and custom similarity
metrics (src/metrics/similarity.py) as recommended by Professor Bade.

This module is kept for reference and potential future use in other
contexts (e.g., RAG for documentation search).

See: docs/phase1_implementation_summary.md for rationale.
"""
import warnings
from qdrant_client import QdrantClient
import os

# Deprecation warning
warnings.warn(
    "qdrant.py is deprecated for road network storage. "
    "Use src/generators/network_storage.py instead.",
    DeprecationWarning,
    stacklevel=2
)

# hint: adapt paths to your needs
# If you've used the provided docker compose file adapt the path like this: QdrantClient("localhost", port=6333)
client = QdrantClient(
    url="https://59b9c848-48ce-49df-b3c2-7d9c3c882bb9.eu-central-1-0.aws.cloud.qdrant.io:6333",
    api_key=os.getenv("QDRANT_API_KEY"),
)

# -*- coding: utf-8 -*-
"""
In this module, you should create tool(s) that the LLM can use to help you in your daily coding tasks.

1. Create a description for your semantic search tool. I should explain clearly what the tool does and how it can be used.
"""
DOCS_FUNCTION = {
    "type": "function",
    "function": {
        "name": "get_chunks_from_python_docs",
        "description": "Use a semantic search to retrieve relevant information chunks from the documentation of the python programming language depending on the user's query. This function is ONLY suitable for questions related to Python and its standard library.",
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "The main topic of the user's query. For example: input: 'How can I change the format of my logs' -> question: 'log formatting'",
                },
            },
            "required": ["question"],
        },
    },
}

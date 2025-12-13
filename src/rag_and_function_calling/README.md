# RAG and function calling exercises
In this folder, you will find all exercises for the RAG and Function Calling module.

Your tasks are:
1. Implement a naive RAG. If you implemented a semantic search in the last module, we would like to encourage you to use it :)
   Else, you can find a ready to use implementation in `/utils/semantic_search.py`.
   - Go to ./rag/rag.py and implement a function that builds a prompt with custom instructions, the user's query and
   relevant chunks.
   - Use ./rag/main.py and  to see how the complete RAG workflow works. Especially pay attention to
   the response quality depending on the quality of the chunks. Maybe you want to rework on the preprocessing at some
   point to possibly increase the search quality.
   ```bash
   cd python_flavour/python_flavour/rag_and_function_calling/rag
   poetry run python main.py
   ```
2. Create a function definition in order to convert your naive RAG into an agentic RAG (= a RAG using function calling)
   and observe the LLM's responses when provided with your function definition.
   - Go to ./function_definition/definitions.py and provide a definition for the LLM to understand the requirements and
   features of your semantic search.
   - Use ./function_definition/main.py to see the LLM's responses in different situations. Play with different questions
   and system prompts.
   ```bash
   cd python_flavour/python_flavour/rag_and_function_calling/function_definition
   poetry run python main.py
   ```
3. Implement an agentic RAG assistant to help you with your daily tasks.
   - Go to ./function_usage/tools.py and implement a callable to get most relevant chunks for a given query.
   - Go to ./function_usage/assistant.py and extend the generate_response method to implement the function calling
   workflow. (Reference: https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/function-calling)
   - Use ./function_usage/main.py to see how the assistant works. Play with different questions and system prompts.
   ```bash
    cd python_flavour/python_flavour/rag_and_function_calling/function_usage
    poetry run python main.py
   ```
4. Extend your assistant!
   - Create new Tools and definitions to extend the assistant's capabilities.
   For instance, you could implement a web search or use the Gitlab search to look for similar code snippets.

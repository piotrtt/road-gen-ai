


# this is only necessary for printing text more nicely in the terminal
from rich.console import Console

from .rag import build_prompt
from ..llm_engine import LLMClient

console = Console()


if __name__ == "__main__":
    # if you don't want your system to be interactive, replace input() with a given question
    user_input = input("Please provide your question: ")
    console.print(":glasses: Your question :glasses:")
    console.print(user_input)
    console.print()

    prompt = build_prompt(user_input)
    console.print(":pen: RAG Prompt :pen:")
    console.print(prompt)
    console.print()

    response = LLMClient().query_structured(prompt)
    answer = response.choices[0].message.content
    # todo: experiment providing an additional system prompt and experimenting on how it affects the response
    console.print(":robot: LLM Answer :robot:")
    console.print(answer)

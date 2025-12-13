# -*- coding: utf-8 -*-
"""
In this module, you should implement an assistant that can call functions based on the LLM's suggestions.
You can find hints for the implementation here: https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/function-calling
"""
import json
from typing import Callable

from pydantic import BaseModel

from ..prompting.dmGPT import call_chat_completion


class Tool(BaseModel):
    name: str
    definition: dict
    fn: Callable


class Assistant:
    def __init__(self, tools: list[Tool]):
        # this is a mapping for you to easily access the definitions and callables
        self.tools = {tool.name: tool for tool in tools}

    def generate_response(self, messages: list) -> str:
        """
        This function generates a response based on the messages, including necessary function calls.
        :param messages: A list of messages
        :return: The LLM response as a string
        """
        response = call_chat_completion(
            messages=messages, tools=[tool.definition for tool in self.tools.values()]
        )

        message = response.choices[0].message

        while message.tool_calls:
            messages.append(message)

            for tool_call in message.tool_calls:
                fn_name = tool_call.function.name
                fn_args = tool_call.function.arguments
                tool = self.tools[fn_name]
                response = tool.fn(**json.loads(fn_args))
                print(f"Calling {tool.name} with arguments {fn_args}")
                print(f"Response: {response}")

                messages.append(
                    {
                        "role": "tool",
                        "name": fn_name,
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(response),
                    }
                )

            response = call_chat_completion(
                messages=messages,
                tools=[tool.definition for tool in self.tools.values()],
            )
            message = response.choices[0].message

        return message.content

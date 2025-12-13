import litellm
import os
import json
from typing import Optional, Any, List, Dict

class LLMClient:
    """
    Wrapper around litellm to abstract interaction with different LLM providers 
    (OpenAI, Anthropic, Google).
    """
    
    def __init__(self, model_name: str = "gpt-4", api_key: Optional[str] = None):
        """
        Initialize the LLM Client.
        
        Args:
            model_name (str): The model identifier. Defaults to env var LLM_MODEL or 'gpt-4'.
            api_key (str, optional): The API key. If not provided, litellm uses os.environ.
        """
        self.model_name = model_name or os.getenv("LLM_MODEL", "gpt-4")
        self.api_key = api_key

        # litellm automatically handles env vars if api_key is not explicitly passed to completion,
        # but we can set it explicitly in os.environ if needed for specific providers not auto-detected
        if api_key:
             # This is a simplification; handling different keys for different providers 
             # usually relies on standard env var names like OPENAI_API_KEY.
             pass

    def query(self, prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
        """
        Sends a prompt to the LLM and returns the response.
        
        Args:
            prompt (str): The user input.
            system_prompt (str): The system context.
            
        Returns:
            str: The raw text response from the model.
        """
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            response = litellm.completion(
                model=self.model_name, 
                messages=messages,
                api_key=self.api_key
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            print(f"[LLM Error] {e}")
            return f"Error querying LLM: {str(e)}"

    def query_structured(self, prompt: str, tools: List[Dict[str, Any]], system_prompt: str = "You are a helpful assistant.") -> Optional[Dict[str, Any]]:
        """
        Uses function calling / tool use to get a structured output.
        
        Args:
            prompt (str): The user input.
            tools (List[Dict]): List of tool definitions (OpenAI format).
            system_prompt (str): System context.
            
        Returns:
             Optional[Dict]: The arguments of the first function call, or None if no call.
        """
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # Helper to force tool use if supported by provider
            tool_choice = "auto" 
            if tools:
                tool_choice = {"type": "function", "function": {"name": tools[0]["function"]["name"]}}

            response = litellm.completion(
                model=self.model_name,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                api_key=self.api_key
            )
            
            message = response['choices'][0]['message']
            if message.get("tool_calls"):
                # Return the arguments of the first tool call parsed as JSON
                tool_call = message["tool_calls"][0]
                function_args = json.loads(tool_call["function"]["arguments"])
                return function_args
                
            return None

        except Exception as e:
            print(f"[LLM Structured Error] {e}")
            return None

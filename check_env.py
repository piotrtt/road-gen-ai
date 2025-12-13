import sys
import os
from dotenv import load_dotenv

# Ensure src is in python path
sys.path.append(os.path.join(os.getcwd(), "src"))

def test_env_loading():
    print("[INFO] Testing Environment Loading...")
    
    # Check if .env exists
    if not os.path.exists(".env"):
        print("[WARNING] .env file not found!")
    else:
        print("[INFO] .env file found.")

    # Load Env
    load_dotenv()
    
    # Verify vars (checking for keys we added to the example)
    
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    openai_model = "gpt-4o-mini"
    anthropic_model = "claude-haiku-4-5-20251001"
    gemini_model = "gemini-2.5-flash"
    
    if openai_key and openai_key != "your_openai_api_key_here":
         print("[INFO] OPENAI_API_KEY is set. The model is {openai_model}")
    else:
         print("[INFO] OPENAI_API_KEY is using default placeholder or not set.")
    
    if anthropic_key and anthropic_key != "your_anthropic_api_key_here":
         print("[INFO] ANTHROPIC_API_KEY is set. The model is {anthropic_model}")
    else:
         print("[INFO] ANTHROPIC_API_KEY is using default placeholder or not set.")
    
    if gemini_key and gemini_key != "your_gemini_api_key_here":
         print("[INFO] GEMINI_API_KEY is set. The model is {gemini_model}")
    else:
         print("[INFO] GEMINI_API_KEY is using default placeholder or not set.")

def test_client_init():
    print("\n[INFO] Testing LLMClient Initialization...")
    try:
        from src.llm_engine.client import LLMClient
        client = LLMClient()
        print(f"[INFO] Initialized Client with model: {client.model_name}")
    except Exception as e:
        print(f"[ERROR] Failed to initialize client: {e}")

def test_llm_connection(model):
    print("\n[INFO] Connecting to LLM...")
    try:
        from src.llm_engine.client import LLMClient
        client = LLMClient(model_name=model)
        response = client.query("Hello! Just say 'Connected to:' and then your model name.")
        print(f"[INFO] LLM Response: {response}")
    except Exception as e:
        print(f"[ERROR] LLM Connection Failed: {e}")

if __name__ == "__main__":
    test_env_loading()
    test_client_init()
    models = ("gpt-4o-mini", "claude-haiku-4-5-20251001", "gemini/gemini-2.5-flash")
    for model in models:
        print(f"\n[INFO] Testing {model}...")
        test_llm_connection(model)

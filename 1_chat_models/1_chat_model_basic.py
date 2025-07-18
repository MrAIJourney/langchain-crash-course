# Chat Model Documents: https://python.langchain.com/v0.2/docs/integrations/chat/
# OpenAI Chat Model Documents: https://python.langchain.com/v0.2/docs/integrations/chat/openai/

from dotenv import load_dotenv
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import PromptTemplate


# Load environment variables from .env
load_dotenv()

# Create ollama model
model = OllamaLLM(
    model="llama3.1",
    temperature=0.1,
    top_p=0.95,
)

# Invoke the model with a message
result = model.invoke("What is 81 divided by 9?")
print("Full result:")
print(result)


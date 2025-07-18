from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_ollama.llms import OllamaLLM

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
model = OllamaLLM(model="llama3.1",
                  temperature=0.1,
                  top_p=0.95) # top_p is a parameter that controls the diversity of the output. A lower value makes the output more focused, while a higher value allows for more diverse responses.

# SystemMessage:
#   Message for priming AI behavior, usually passed in as the first of a sequenc of input messages.
# HumanMessagse:
#   Message from a human to the AI model.
messages = [
    SystemMessage(content="Solve the following math problems, Explain it for a 7 year old."),
    HumanMessage(content="What is 81 divided by 9?"),
]

# Invoke the model with messages
result = model.invoke(messages)
print(f"Answer from AI: {result}")


# AIMessage:
#   Message from an AI. you can keep track of the AI's responses in a conversation.
messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 81 divided by 9?"),
    AIMessage(content="81 divided by 9 is 9."), # AI's response to the previous human message
    HumanMessage(content="What is 10 times 5?"), # new human message
]

# Invoke the model with messages
result = model.invoke(messages)
print(f"Answer from AI: {result.content}")

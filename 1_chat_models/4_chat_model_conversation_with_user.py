from dotenv import load_dotenv
from langchain_google_genai import  ChatGoogleGenerativeAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.8)


chat_history = []  # Use a list to store messages

# Set an initial system message (optional)
system_message = SystemMessage(content="Yor are a comedian AI that answers questions with humor and wit. ")
chat_history.append(system_message)  # Add system message to chat history

# Chat loop
while True:
    query = input("You(type 'exit' to quit): ")
    if query.lower() == "exit":
        break
    chat_history.append(HumanMessage(content=query))  # Add user message

    # Get AI response using history
    result = model.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))  # Add AI message

    print(f"AI: {response}")


print("---- Message History ----")
print(chat_history)

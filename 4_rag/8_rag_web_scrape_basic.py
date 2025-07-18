import os

from dotenv import load_dotenv
from langchain.chains import history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load environment variables from .env
load_dotenv()

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
db_dir = os.path.join(current_dir, "db")
persistent_directory = os.path.join(db_dir, "chroma_db_mraijourney")

# Step 1: Scrape the content from mraijourney.com using WebBaseLoader
# WebBaseLoader loads web pages and extracts their content
urls = ["https://www.mraijourney.com/"]

# Create a loader for web content
loader = WebBaseLoader(urls)
documents = loader.load()

# Step 2: Split the scraped content into chunks
# CharacterTextSplitter splits the text into smaller chunks
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# Display information about the split documents
# print("\n--- Document Chunks Information ---")
# print(f"Number of document chunks: {len(docs)}")
# print(f"Sample chunk:\n{docs[0].page_content}\n")

# Step 3: Create embeddings for the document chunks
# OpenAIEmbeddings turns text into numerical vectors that capture semantic meaning
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Step 4: Create and persist the vector store with the embeddings
# Chroma stores the embeddings for efficient searching
if not os.path.exists(persistent_directory):
    print(f"\n--- Creating vector store in {persistent_directory} ---")
    db = Chroma.from_documents(docs, embeddings, persist_directory=persistent_directory)
    print(f"--- Finished creating vector store in {persistent_directory} ---")
else:
    print(f"Vector store {persistent_directory} already exists. No need to initialize.")
    db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Step 5: Query the vector store
# Create a retriever for querying the vector store
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)
# Create an ollama model
model = OllamaLLM(model="llama3.1")


# Display the relevant results with metadata
system_prompt_contextualize = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)
# Define the messages for the model
contextualize_prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_contextualize),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
# Create a history-aware retriever that uses the LLM to help reformulate the question based on chat history
history_aware_retriever = create_history_aware_retriever(
     model,
    retriever,
    contextualize_prompt_template,

)

# Create a prompt template for the question-aware chain
system_prompt_question_aware = (
    "Given the retrieved documents, answer the question based on the content of those documents. "
    "If the answer is not found in the documents, respond with 'I'm not sure'."
    "\n\n"
    "{context}" # This will be replaced with the retrieved documents
)
question_aware_template = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt_question_aware),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a question-aware chain that combines the retrieved documents and the question
question_aware_chain = create_stuff_documents_chain(model, question_aware_template)

# Create a retrieval chain that combines the history-aware retriever and the question-aware chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_aware_chain)

# Step 6: Start a conversational loop to ask questions
chat_history = []
while True:
    query = input("\nEnter your question (or type 'exit' to quit): ")
    if query.lower() == 'exit':
        break
    print("Starting the rag chain with the query...")
    result = rag_chain.invoke({
        "input": query,
        "chat_history": chat_history
    })
    # display the result
    print(f"\n--- Answer ---\n{result['answer']}\n")
    # append the question and answer to chat history
    chat_history.append(HumanMessage(content=query))
    chat_history.append(SystemMessage(content=result['answer']))

# # Define the user's question
# query = "Who is Mohammad Najafi and what is his background in AI and ML?"
#
# # Retrieve relevant documents based on the query
# relevant_docs = retriever.invoke(query)
# print("\n--- Relevant Documents ---")
# for i, doc in enumerate(relevant_docs, 1):
#     print(f"Document {i}:\n{doc.page_content}\n")
#     if doc.metadata:
#         print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")

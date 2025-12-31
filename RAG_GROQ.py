# main.py

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0,
    groq_api_key=os.getenv("GROQ_API_KEY"))

print("LLM loaded: llama3 via groq")

# -----------------------------
# 2. Load documents (safe from encoding errors)
# -----------------------------
loader = DirectoryLoader(
    r"D:\Codebasics\GenAI\Vector Database\new_articles",
    glob="**/*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8", "autodetect_encoding": True},
    silent_errors=True,
)

documents = loader.load()
print(f"Loaded {len(documents)} documents.")

if len(documents) == 0:
    raise ValueError("No documents were loaded. Check your folder path.")

# -----------------------------
# 3. Split into chunks
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)

texts = text_splitter.split_documents(documents)
print(f"Split into {len(texts)} chunks.")

# -----------------------------
# 4. Embeddings (local & free)
# -----------------------------
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# -----------------------------
# 5. Create / Load persistent Chroma database
# -----------------------------
persist_directory = r"D:\Codebasics\GenAI\Vector Database\db"

# === RUN THIS BLOCK ONCE TO BUILD THE DATABASE ===
vector_db = Chroma.from_documents(
    documents=texts,
    embedding=embedding_function,
    persist_directory=persist_directory
)
print("Vector database created and saved to disk.")

# === AFTER FIRST RUN: COMMENT THE ABOVE AND UNCOMMENT BELOW FOR FASTER START ===
# vector_db = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)
# print("Loaded existing vector database from disk.")

# -----------------------------
# 6. Create retriever
# -----------------------------
retriever = vector_db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 6}  # Get top 6 relevant chunks
)

# -----------------------------
# 7. Define a clear RAG prompt
# -----------------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer the user's question using only the information from the context below. "
               "If the context doesn't contain relevant information, say 'I don't know based on the provided articles.'\n\n"
               "Context:\n{context}"),
    ("human", "{question}")
])

# Helper to format retrieved documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# -----------------------------
# 8. Build the RAG chain
# -----------------------------
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# -----------------------------
# 9. Test with sample questions
# -----------------------------
queries = [
    "How much money did Microsoft raise?",
    "What is Pando and how much funding did it get?",
    "Which companies received funding in the articles?",
    "Summarize the main news about AI startups."
]

print("\n" + "="*80)
print("RAG SYSTEM READY – ASKING QUESTIONS (100% free & local)")
print("="*80)

for query in queries:
    print(f"\nQuestion: {query}")
    print("-" * 60)
    answer = rag_chain.invoke(query)
    print(f"Answer:\n{answer.strip()}\n")

# -----------------------------
# 10. Interactive mode (keep asking questions)
# -----------------------------
print("Interactive mode started! Type 'exit' to quit.\n")
while True:
    user_query = input("Your question: ").strip()
    if user_query.lower() in ["exit", "quit", "bye"]:
        print("Goodbye!")
        break
    if user_query:
        print("Thinking...\n")
        response = rag_chain.invoke(user_query)
        print(f"Answer:\n{response.strip()}\n")
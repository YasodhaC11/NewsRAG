# NewsRAG – A Fast, Accurate RAG System Powered by Groq

NewsRAG is a complete Retrieval-Augmented Generation (RAG) application that allows you to ask natural-language questions about a collection of 21 tech news articles (focused on AI startups, funding, and big tech investments) and receive highly accurate, well-reasoned answers — all powered by the Groq cloud API for lightning-fast, high-quality inference.
This project demonstrates a modern, production-style RAG pipeline using only open-source tools for the local components and Groq's free tier for the language model — achieving near-GPT-4 level performance at zero cost and blazing speed.
## Key Features

-Accurate & Reliable Answers
-Groq runs powerful models like Llama 3 8B/70B, resulting in minimal hallucinations and strict adherence to the provided context.
- Ultra-Fast Responses
Groq's custom LPU (Language Processing Unit) hardware delivers inference speeds often 5–10x faster than traditional GPU providers.
Local Processing for Documents
All document loading, chunking, embedding, and vector search happen entirely on your machine — only the question + relevant context is sent to Groq.
Persistent Vector Database
Uses Chroma to store embeddings — built once, reused instantly on future runs.
Interactive Terminal Interface
Ask unlimited questions with real-time responses.
Free to Use
Leverages Groq's generous free tier (no credit card required).

## How It Works

Document Loading
Reads all .txt files from the new_articles/ folder (handles encoding safely).
Text Chunking
Splits articles into overlapping 1000-character chunks for better retrieval accuracy.
Local Embeddings
Uses Hugging Face's all-MiniLM-L6-v2 model (fully offline) to create vector representations.
Vector Store
Stores embeddings in a persistent Chroma database (db/ folder).
Retrieval
For each question, finds the top 6 most relevant document chunks using similarity search.
Generation with Groq
Sends the retrieved context + question to Groq's Llama 3 model via API → generates a grounded, factual response.

🚀 Why Groq Makes This Project Stand Out

Speed: Responses appear almost instantly (often under 1 second).
Intelligence: Llama 3 (especially 70B) provides deep reasoning, nuance, and precision far beyond small local models.
Accuracy: Strong context adherence → correctly identifies when information is missing (e.g., "Microsoft did not raise money — it invested in OpenAI").
Free Tier Friendly: Perfect for learning, prototyping, and personal projects.

📊 Example Q&A (Real Output from This Project)

Q: How much money did Microsoft raise?
A: Microsoft did not raise money. It invested around $10 billion in OpenAI.
Q: What is Pando and how much funding did it get?
A: Pando is a startup developing fulfillment management technologies. It raised $30 million in Series B, total $45 million.
Q: Which companies received funding?
A: Primarily OpenAI (backed by Microsoft, Sequoia, a16z, etc.) — correctly distinguishes investors from recipients.
Q: Career opportunities in AI startups?
A: "I don't know based on the provided articles" — honest when info isn't present.

🔧 Tech Stack

LangChain – RAG pipeline orchestration
Chroma – Local persistent vector database
HuggingFace Embeddings – Local sentence embeddings (all-MiniLM-L6-v2)
Groq API – Cloud LLM inference (Llama 3 8B/70B)
Python – Core implementation

🎯 Perfect For

Learning modern RAG architecture
Building personal knowledge bases (swap in your own documents!)
Portfolio project showcasing LangChain + cloud inference
Comparing local vs. cloud LLM performance

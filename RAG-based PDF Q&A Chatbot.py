"""
RAG-based PDF Q&A Chatbot
Uses LangChain + Google Gemini + FAISS vector store
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
GEMINI_MODEL   = "gemini-1.5-flash"
EMBED_MODEL    = "models/embedding-001"
CHUNK_SIZE     = 1000
CHUNK_OVERLAP  = 150
TOP_K_DOCS     = 4
MEMORY_WINDOW  = 6          # how many last exchanges to keep in memory

# ── Custom prompt ─────────────────────────────────────────────────────────────
QA_PROMPT_TEMPLATE = """You are a helpful and precise document assistant. 
Answer the user's question using ONLY the provided context from the document.
If the answer is not in the context, say "I couldn't find that in the document."
Never make up information.

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Answer:"""

QA_PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=QA_PROMPT_TEMPLATE,
)


class RAGChatbot:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            temperature=0.2,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBED_MODEL,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
        )
        self.vector_store = None
        self.chain = None
        self.current_pdf = None

    # ── Document ingestion ────────────────────────────────────────────────────
    def load_pdf(self, pdf_path: str) -> int:
        """Load a PDF, split it, embed it, and build the retriever. Returns chunk count."""
        print(f"\n📄 Loading: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ".", " "],
        )
        chunks = splitter.split_documents(documents)
        print(f"✂️  Split into {len(chunks)} chunks")

        print("🔢 Embedding chunks...")
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)

        self._build_chain()
        self.current_pdf = os.path.basename(pdf_path)
        print(f"✅ Ready! Ask me anything about '{self.current_pdf}'\n")
        return len(chunks)

    def save_index(self, path: str = "faiss_index"):
        """Persist the FAISS index to disk so you don't re-embed every run."""
        if self.vector_store:
            self.vector_store.save_local(path)
            print(f"💾 Index saved to '{path}/'")

    def load_index(self, path: str = "faiss_index", pdf_name: str = "saved document"):
        """Load a previously saved FAISS index."""
        self.vector_store = FAISS.load_local(
            path, self.embeddings, allow_dangerous_deserialization=True
        )
        self._build_chain()
        self.current_pdf = pdf_name
        print(f"✅ Loaded index from '{path}/' — ready to chat!\n")

    # ── Chain assembly ────────────────────────────────────────────────────────
    def _build_chain(self):
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K_DOCS},
        )
        memory = ConversationBufferWindowMemory(
            k=MEMORY_WINDOW,
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
        )
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=memory,
            combine_docs_chain_kwargs={"prompt": QA_PROMPT},
            return_source_documents=True,
            verbose=False,
        )

    # ── Chat ──────────────────────────────────────────────────────────────────
    def chat(self, question: str) -> dict:
        """Ask a question. Returns answer + source page numbers."""
        if not self.chain:
            return {"answer": "⚠️ No document loaded. Use load_pdf() first.", "sources": []}

        result = self.chain.invoke({"question": question})
        sources = sorted(
            set(doc.metadata.get("page", "?") + 1 for doc in result["source_documents"])
        )
        return {"answer": result["answer"], "sources": sources}

    def reset_memory(self):
        """Clear conversation history without reloading the document."""
        if self.chain:
            self.chain.memory.clear()
            print("🧹 Memory cleared.")


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("        📚 RAG PDF Chatbot  |  LangChain + Gemini")
    print("=" * 60)

    bot = RAGChatbot()

    # ── Load document ─────────────────────────────────────────────────────
    while True:
        pdf_path = input("\nEnter path to your PDF file: ").strip().strip('"')
        if os.path.isfile(pdf_path) and pdf_path.lower().endswith(".pdf"):
            bot.load_pdf(pdf_path)
            break
        print("❌ File not found or not a PDF. Try again.")

    # ── Chat loop ─────────────────────────────────────────────────────────
    print("Commands: 'quit' to exit | 'reset' to clear memory | 'save' to save index\n")
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n👋 Goodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("👋 Goodbye!")
            break
        if question.lower() == "reset":
            bot.reset_memory()
            continue
        if question.lower() == "save":
            bot.save_index()
            continue

        response = bot.chat(question)
        print(f"\nBot: {response['answer']}")
        if response["sources"]:
            print(f"     📌 Sources: page(s) {response['sources']}")
        print()


if __name__ == "__main__":
    main()
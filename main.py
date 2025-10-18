import os
import glob
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import ObsidianLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def create_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"✅ Split into {len(split_docs)} chunks.")
    return split_docs

def build_vectorstore(docs, persist_path):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(docs, embedding=embeddings)
    vectorstore.save_local(persist_path)
    print(f"✅ FAISS index built and saved to: {persist_path}")

def load_vectorstore(index_path="notes_index"):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

def query_loop(db):
    print("\n✅ Ready! Test similarity of query text to notes")

    while True:
        query = input("You: ").strip()
        if query.lower() in {"exit", "quit"}:
            break

        results = db.similarity_search(query, k=3)
        print("\nTop results:")
        for i, r in enumerate(results, 1):
            print(f"\n[{i}] From: {r.metadata['source']}")
            print("-" * 60)
            print(r.page_content[:500].strip(), "...\n")
        print("=" * 80)

def main():
    notes_dir = input("Enter location of notes folder: ")
    index_path = "notes_index"

    if not os.path.exists(notes_dir):
        print(f"Directory does not exist")
        return

    if os.path.exists(index_path):
        print("Loading existing vector index...")
        db = load_vectorstore(index_path)
    else:
        print("Building new vector index from notes...")
        obs_loader = ObsidianLoader(notes_dir)
        notes = obs_loader.load()
        chunks = create_chunks(notes)
        build_vectorstore(chunks, index_path)
        db = load_vectorstore(index_path)
    
    query_loop(db)

if __name__ == "__main__":
    main()
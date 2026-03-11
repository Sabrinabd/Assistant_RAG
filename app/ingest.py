from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os


PDF_FOLDER = "data"
DB_FOLDER  = "vectorstore"

def ingest_pdfs():
    documents = []

    # Lire tous les PDFs dans le dossier data/
    for fichier in os.listdir(PDF_FOLDER):
        if fichier.endswith(".pdf"):
            print(f" Lecture de : {fichier}")
            loader = PyPDFLoader(os.path.join(PDF_FOLDER, fichier))
            documents.extend(loader.load())

    # Découper en petits morceaux
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"{len(chunks)} morceaux créés")

    # Créer les embeddings et stocker dans ChromaDB
    print(" Création des embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_FOLDER
    )
    print(" Base vectorielle créée avec succès !")

if __name__ == "__main__":
    ingest_pdfs()
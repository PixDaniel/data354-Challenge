import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

def create_vector_store_from_pdf(pdf_path: str, save_path: str = "faiss_index"):
    """
    Crée et sauvegarde une base de données vectorielle FAISS à partir d'un fichier PDF.

    Args:
        pdf_path (str): Le chemin vers le fichier PDF.
        save_path (str): Le chemin où sauvegarder l'index FAISS.
    """
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY non trouvée dans le fichier .env")

    # 1. Charger le document PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # 2. Découper le document en chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    # 3. Créer les embeddings avec Gemini
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)

    # 4. Créer et sauvegarder la base de données vectorielle FAISS
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(save_path)
    print(f"La base de données vectorielle a été créée et sauvegardée à l'emplacement : {save_path}")

if __name__ == '__main__':
    # Assurez-vous que le PDF est dans le bon dossier
    pdf_file_path = "src/data/Code du Travail Ivoirien 2021.pdf"
    if os.path.exists(pdf_file_path):
        create_vector_store_from_pdf(pdf_file_path, save_path="src/data/faiss_index")
    else:
        print(f"Erreur : Le fichier {pdf_file_path} n'a pas été trouvé.")
        print("Veuillez le placer dans le dossier 'src/data' avant de lancer le script.") 
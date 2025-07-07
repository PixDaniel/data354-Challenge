import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

class BaselineRAG:
    """
    Implémentation d'un pipeline RAG de base (Baseline).
    """
    def __init__(self, vector_store_path="src/data/faiss_index"):
        load_dotenv()
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY non trouvée dans le fichier .env")

        self.vector_store = self._load_vector_store(vector_store_path)
        self.llm = GoogleGenerativeAI(model="gemini-1.5-pro-latest", google_api_key=self.api_key)
        self.retriever = self.vector_store.as_retriever()
        self.chain = self._build_chain()

    def _load_vector_store(self, path: str) -> FAISS:
        """Charge la base de données vectorielle FAISS."""
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=self.api_key)
        return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)

    def _build_chain(self):
        """Construit la chaîne de traitement RAG."""
        template = """
        Vous êtes un assistant spécialisé dans le droit du travail en Côte d'Ivoire.
        Répondez à la question suivante en vous basant uniquement sur le contexte fourni.
        Si l'information n'est pas dans le contexte, dites "Je ne trouve pas l'information dans le document."
        
        Contexte:
        {context}
        
        Question:
        {question}
        
        Réponse:
        """
        prompt = ChatPromptTemplate.from_template(template)

        return (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

    def ask(self, question: str) -> str:
        """Pose une question au pipeline RAG et retourne la réponse."""
        return self.chain.invoke(question)

    def ask_for_eval(self, question: str) -> dict:
        """
        Pose une question et retourne la réponse ainsi que le contexte récupéré,
        pour l'évaluation.
        """
        retrieved_docs = self.retriever.invoke(question)
        answer = self.chain.invoke(question)
        contexts = [doc.page_content for doc in retrieved_docs]
        return {"answer": answer, "contexts": contexts}

if __name__ == '__main__':
    rag_agent = BaselineRAG()
    
    # Exemple de question
    question = "Quelles sont les conditions pour un licenciement pour motif économique ?"
    
    print(f"Question : {question}")
    response = rag_agent.ask(question)
    print(f"Réponse : {response}")

    question_2 = "Quelle est la durée du congé de maternité ?"
    print(f"\nQuestion : {question_2}")
    response_2 = rag_agent.ask(question_2)
    print(f"Réponse : {response_2}") 
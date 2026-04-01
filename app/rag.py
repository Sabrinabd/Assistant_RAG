from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

DB_FOLDER = "vectorstore"

def load_rag_chain():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = Chroma(
        persist_directory=DB_FOLDER,
        embedding_function=embeddings
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = Ollama(model="mistral")

    prompt = PromptTemplate.from_template("""
    Utilise uniquement le contexte suivant pour répondre à la question.
    Si tu ne trouves pas la réponse, dis "Je ne sais pas".

    Contexte : {context}
    Question : {question}
    Réponse :
    """)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

def ask_question(question: str) -> str:
    chain = load_rag_chain()
    return chain.invoke(question)

if __name__ == "__main__":
    print(" Assistant RAG prêt !")
    while True:
        question = input("\n Ta question : ")
        if question.lower() == "quitter":
            break
        reponse = ask_question(question)
        print(f"\n Réponse : {reponse}")
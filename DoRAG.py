import os
import sys
import json
from datetime import datetime
from collections import deque
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
import warnings

warnings.filterwarnings("ignore", message="cumsum_out_mps supported by MPS on MacOS 13+")

def get_valid_pdf_path():
    while True:
        pdf_path = input("Enter the path to your PDF file: ").strip()
        if os.path.isfile(pdf_path) and pdf_path.lower().endswith('.pdf'):
            return pdf_path
        else:
            print("Invalid file path or not a PDF file. Please try again.")

def load_and_split_pdf(pdf_path):
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        print(f'Loaded {len(pages)} pages')
    except Exception as e:
        print(f"Error loading the PDF file: {e}")
        sys.exit(1)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(pages)
    print(f"Created {len(splits)} splits from the PDF")
    return splits

def setup_vectorstore(splits):
    vectorstore = Chroma.from_documents(documents=splits, embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))
    retriever = vectorstore.as_retriever()
    return retriever

def get_prompts():
    system_prompt = """You are an AI assistant named DoRAG tasked with answering questions based on the provided context. 
    Always strive to give accurate and helpful responses. You will be provided with conversation history for context along with PDF data.
    Use these resources and provide the best responses possible."""

    question_prompt_template = """Conversation history:
{history}

Current context and question:
{context}

Question: {question}

Answer:"""
    return system_prompt, question_prompt_template

def setup_rag_chain(retriever, system_prompt, question_prompt_template):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", question_prompt_template),
    ])

    llm = Ollama(model='llama3')

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def format_history(history):
        return "\n".join([f"Human: {h['question']}\nAI: {h['answer']}" for h in history])

    rag_chain = (
        {
            "history": lambda x: format_history(x["history"]),
            "context": lambda x: format_docs(retriever.invoke(x["question"])),
            "question": lambda x: x["question"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain, format_docs

def main():
    pdf_path = get_valid_pdf_path()
    splits = load_and_split_pdf(pdf_path)
    retriever = setup_vectorstore(splits)
    system_prompt, question_prompt_template = get_prompts()
    rag_chain, format_docs = setup_rag_chain(retriever, system_prompt, question_prompt_template)
    conversation_history = deque(maxlen=10)  # Adjust maxlen as needed

    while True:
        user_question = input("\nEnter your question about the PDF content (or 'quit' to exit): ")
        
        if user_question.lower() == 'quit':
            print("Goodbye!")
            break

        try:
            context = format_docs(retriever.invoke(user_question))
            print("\nRetrieved context:")
            print(context[:500] + "..." if len(context) > 500 else context)
            
            print("\nUser question:")
            print(user_question)
            
            result = rag_chain.invoke({"question": user_question, "history": list(conversation_history)})
            print("\nAnswer:")
            print(result)

            conversation_history.append({"question": user_question, "answer": result})
        except Exception as e:
            print(f"An error occurred while processing your question: {e}")
            print("Please ensure that Ollama is installed and running.")
            print("You can start Ollama by running the 'ollama' command in a terminal.")
            print("If the problem persists, check your firewall settings or try specifying the Ollama URL explicitly in the code.")
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
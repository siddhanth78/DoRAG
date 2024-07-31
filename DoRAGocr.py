import os
import sys
from collections import deque
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
import warnings
from PIL import Image
import pytesseract
from pdf2image import convert_from_path

warnings.filterwarnings("ignore", message="cumsum_out_mps supported by MPS on MacOS 13+")

def get_valid_file_path():
    while True:
        file_path = input("Enter the path to your PDF or image file: ").strip()
        if os.path.isfile(file_path) and file_path.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg')):
            return file_path
        else:
            print("Invalid file path or unsupported file type. Please try again.")

def extract_text_from_image(image):
    return pytesseract.image_to_string(image)

def load_and_split_file(file_path):
    try:
        if file_path.lower().endswith('.pdf'):
            # Convert PDF to images
            images = convert_from_path(file_path)
            print(f'Converted PDF to {len(images)} images')
            
            pages = []
            for i, image in enumerate(images):
                text = extract_text_from_image(image)
                if text.strip():
                    pages.append(Document(page_content=text, metadata={"source": file_path, "page": i+1}))
                else:
                    print(f"Warning: No text extracted from page {i+1}")
            
            if not pages:
                raise Exception("No content extracted from PDF")
            
            print(f'Extracted text from {len(pages)} pages')
        else:  # Single image file
            image = Image.open(file_path)
            text = extract_text_from_image(image)
            if not text.strip():
                raise Exception("No text extracted from image")
            pages = [Document(page_content=text, metadata={"source": file_path})]
            print(f'Extracted text from image')
    except Exception as e:
        print(f"Error loading the file: {e}")
        sys.exit(1)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(pages)
    print(f"Created {len(splits)} splits from the file")
    return splits

def setup_vectorstore(splits):
    vectorstore = Chroma.from_documents(documents=splits, embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))
    retriever = vectorstore.as_retriever()
    return retriever

def get_prompts():
    system_prompt = """You are an AI assistant named DoRAG tasked with answering questions based on the provided context. 
    Always strive to give accurate and helpful responses. You will be provided with conversation history for context along with file data.
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
    file_path = get_valid_file_path()
    splits = load_and_split_file(file_path)
    retriever = setup_vectorstore(splits)
    system_prompt, question_prompt_template = get_prompts()
    rag_chain, format_docs = setup_rag_chain(retriever, system_prompt, question_prompt_template)
    conversation_history = deque(maxlen=10)  # Adjust maxlen as needed

    while True:
        user_question = input("\nEnter your question about the file content (or 'quit' to exit): ")
        
        if user_question.lower() == 'quit':
            print("Goodbye!")
            break

        try:
            context = format_docs(retriever.invoke(user_question))
            print("\nRetrieved context:")
            print(context[:500] + "..." if len(context) > 500 else context)
            
            print("\nUser question:")
            print(user_question)
            
            print("\nAnswer:")
            for result in rag_chain.stream({"question": user_question, "history": list(conversation_history)}):
                print(result, end='', flush=True)

            conversation_history.append({"question": user_question, "answer": result})
        except Exception as e:
            print(f"An error occurred while processing your question: {e}")
            print("Please ensure that Ollama is installed and running.")
            print("You can start Ollama by running the 'ollama' command in a terminal.")
            print("If the problem persists, check your firewall settings or try specifying the Ollama URL explicitly in the code.")
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()

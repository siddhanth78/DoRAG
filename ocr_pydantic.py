import os
import sys
from collections import deque
import json
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
import warnings
from PIL import Image
import pytesseract
from pdf2image import convert_from_path
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
from langchain_core.output_parsers import JsonOutputParser

warnings.filterwarnings("ignore", message="cumsum_out_mps supported by MPS on MacOS 13+")

class DocumentFields(BaseModel):
    fields: Optional[Dict[str, Any]] = Field(default_factory=dict, description="A dictionary of field names and their corresponding values")
    content: Optional[str] = Field(default=None, description="The content of the response if not in field format")

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

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
    splits = text_splitter.split_documents(pages)
    print(f"Created {len(splits)} splits from the file")
    return splits

def setup_vectorstore(splits):
    vectorstore = Chroma.from_documents(documents=splits, embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'))
    return vectorstore

def get_prompts():
    system_prompt = """You are an AI assistant tasked with answering questions and extracting fields from forms based on the provided context.
    If the question asks for specific fields, return a JSON object with a 'fields' key containing field names and their corresponding values.
    If the question is more general, provide a regular text response.
    Always strive to give accurate and helpful responses based on the provided context."""

    question_prompt_template = """
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

    def parse_output(output: str) -> DocumentFields:
        try:
            # Try to parse as JSON first
            json_output = JsonOutputParser().parse(output)
            return DocumentFields(**json_output)
        except:
            # If JSON parsing fails, treat it as plain text
            return DocumentFields(content=output)

    rag_chain = (
        {
            "context": lambda x: format_docs(retriever.invoke(x["question"])),
            "question": lambda x: x["question"],
        }
        | prompt
        | llm
        | StrOutputParser()
        | parse_output
    )
    return rag_chain, format_docs

def main():
    file_path = get_valid_file_path()
    splits = load_and_split_file(file_path)
    vectorstore = setup_vectorstore(splits)
    system_prompt, question_prompt_template = get_prompts()
    rag_chain, format_docs = setup_rag_chain(vectorstore.as_retriever(), system_prompt, question_prompt_template)
    conversation_history = deque(maxlen=10)  # Adjust maxlen as needed

    while True:
        user_question = input("\nEnter your question about the file content (or 'quit' to exit): ")
        
        if user_question.lower() == 'quit':
            print("Goodbye!")
            break

        try:
            # Perform similarity search
            similar_docs = vectorstore.similarity_search(user_question, k=10)  # Retrieve top 10 similar documents
            context = format_docs(similar_docs)
            
            #print("\nRetrieved context:")
            #print(context[:500] + "..." if len(context) > 500 else context)
            
            print("\nUser question:")
            print(user_question)
            
            print("\nAnswer:")
            result = rag_chain.invoke({"question": user_question})
            
            if result.fields:
                print("Extracted fields:")
                print(json.dumps(result.fields, indent=2))
            elif result.content:
                print("Response:")
                print(result.content)
            else:
                print("No response generated.")

            conversation_history.append({"question": user_question, "answer": result.model_dump()})
        except Exception as e:
            print(f"An error occurred while processing your question: {e}")
            print("Please ensure that Ollama is installed and running.")
            print("You can start Ollama by running the 'ollama' command in a terminal.")
            print("If the problem persists, check your firewall settings or try specifying the Ollama URL explicitly in the code.")
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()

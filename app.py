import streamlit as st # type: ignore
import langchain_community
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from PyPDF2 import PdfReader # type: ignore
from docx import Document # type: ignore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

# Function to read files and return text
def read_file(file):
    file_type = file.type
    text = ""
    
    if file_type == "application/pdf":
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    
    elif file_type == "text/plain":
        text = file.read().decode("utf-8")
    
    return text

# Define the prompt template
template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know. 
Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer.
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

# Streamlit App
def main():
    st.title("Docu Pal - Chat with Your Documents 💬")
 
    # Create an empty placeholder
    placeholder = st.empty()
    # File upload
    uploaded_file = st.file_uploader("Upload your document 📄", type=["pdf", "docx", "txt"])
    # Check if a file has been uploaded
    if uploaded_file:
        with placeholder.container():
            # Extract text from the uploaded file
            doc_text = read_file(uploaded_file)
 
            # Initialize vector store and embeddings
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key='XXXXXXXXXXX')
            vector_store = FAISS.from_texts([doc_text], embedding=embeddings)
 
            # Set up the QA chain with the retriever
            qa_chain = RetrievalQA.from_chain_type(
                llm=ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key='XXXXXXXXXXXXXXX'),
                retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
            )
 
            # Input question
            user_question = st.text_input("Ask me a question ❓:")
            if user_question:
                result = qa_chain({"query": user_question})
                st.write(result['result'])  # Display the answer
 
    else:
        # Display a message to upload the file
        placeholder.text("Please upload a document to start.")
 
if __name__ == "__main__":
    main()

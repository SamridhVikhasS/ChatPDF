import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
import os
import tempfile

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Configure Google Gemini API key from environment variables
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Retrieve login credentials from Streamlit secrets
#VALID_USERNAME = st.secrets["USER_NAME"]
#VALID_PASSWORD = st.secrets["PASSWORD"]

def get_pdf_text(pdf_path):
    """Extracts text from a PDF document."""
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    """Splits the extracted text into manageable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store(text_chunks):
    """Creates an in-memory vector store from text chunks."""
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    return vector_store

def generate_response(context, question):
    """Generates a response to a question based on the provided context."""
    prompt_template = """
    Answer the question as detailed as possible from the provided context. Make sure to provide all the details. If the answer is not in the provided context, just say, "The answer is not in the provided context." Do not provide the wrong answer.
    and make the answer more visually attractive, structured and precise.
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    
    prompt = prompt_template.format(context=context, question=question)
    
    # Initialize the model with the correct name
    model = genai.GenerativeModel('gemini-1.0-pro')
    
    # Start a chat session and send the message
    chat_session = model.start_chat(
        history=[]
    )
    
    response = chat_session.send_message(prompt)
    return response.text

def user_input(user_question, vector_store):
    """Handles user input and generates a response based on the in-memory vector store."""
    try:
        # Perform similarity search
        docs = vector_store.similarity_search(user_question)
    except Exception as e:
        st.error(f"Failed to perform similarity search: {e}")
        return "Error: Unable to perform the similarity search."

    # Extracting text from the retrieved documents
    context = "\n".join([doc.page_content for doc in docs])

    try:
        # Generate response using the custom function
        response = generate_response(context, user_question)
    except Exception as e:
        st.error(f"Failed to generate response: {e}")
        return "Error: Unable to generate a response."

    return response

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="Chat PDF", layout="wide")

    # Single Column Layout
    st.header("Chat PDF")

    #if 'logged_in' not in st.session_state:
     #   st.session_state.logged_in = False

    #if not st.session_state.logged_in:
     #   st.subheader("Login")
      #  username = st.text_input("Username")
       # password = st.text_input("Password", type="password")

        #if st.button("Login"):
         #   if username == VALID_USERNAME and password == VALID_PASSWORD:
          #      st.session_state.logged_in = True
           #     st.session_state.image_index = 0  # Initialize image index
            #    st.success("Login successful!")
            #else:
             #   st.error("Invalid username or password")
        #return  # Exit early to not show the main app while not logged in"""

    # PDF Upload Section
    pdf_file = st.file_uploader("Upload PDF", type="pdf")
    if pdf_file:
        # Extract text from the PDF and create an in-memory vector store
        pdf_text = get_pdf_text(pdf_file)
        text_chunks = get_text_chunks(pdf_text)
        vector_store = create_vector_store(text_chunks)
        st.session_state.vector_store = vector_store  # Store in session state

    # Query Section
    user_question = st.text_input("Ask a Question")

    if user_question:
        if 'vector_store' in st.session_state:
            vector_store = st.session_state.vector_store
            response = user_input(user_question, vector_store)
            st.write("## Reply:")
            st.write(response)
        else:
            st.write("Please upload a PDF to begin.")

if __name__ == "__main__":
    main()

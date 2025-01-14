import streamlit as st
from htmlTemplate import get_html_template
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain.vectorstores import Pinecone as LangchainPinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize Pinecone with new method
try:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    PINECONE_INDEX_NAME = "pdf-chat-index"
    
    # Check if index exists
    existing_indexes = pc.list_indexes().names()
    if PINECONE_INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=768,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    pinecone_available = True
    print("Pinecone initialized successfully")
except Exception as e:
    print(f"Pinecone initialization failed: {e}")
    pinecone_available = False

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_hybrid_vector_store(text_chunks):
    # Create embeddings with explicit API key
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    
    # Always create FAISS index as backup
    faiss_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    faiss_store.save_local("faiss_index")
    
    # Try to store in Pinecone if available
    if pinecone_available:
        try:
            index = pc.Index(PINECONE_INDEX_NAME)
            pinecone_store = LangchainPinecone.from_texts(
                texts=text_chunks,
                embedding=embeddings,
                index_name=PINECONE_INDEX_NAME
            )
            return {"faiss": faiss_store, "pinecone": pinecone_store, "primary": "pinecone"}
        except Exception as e:
            print(f"Pinecone storage failed: {e}")
            return {"faiss": faiss_store, "primary": "faiss"}
    else:
        return {"faiss": faiss_store, "primary": "faiss"}

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the context, respond with "Answer is not available in the context."

    Context:
    {context}
    
    Question:
    {question}

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    
    # Try Pinecone first if available
    if pinecone_available:
        try:
            pinecone_store = LangchainPinecone.from_existing_index(
                index_name=PINECONE_INDEX_NAME,
                embedding=embeddings
            )
            docs = pinecone_store.similarity_search(user_question)
            st.session_state["vector_store_used"] = "Pinecone"
        except Exception as e:
            print(f"Pinecone search failed: {e}")
            # Fallback to FAISS
            faiss_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = faiss_store.similarity_search(user_question)
            st.session_state["vector_store_used"] = "FAISS (fallback)"
    else:
        # Use FAISS directly
        faiss_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = faiss_store.similarity_search(user_question)
        st.session_state["vector_store_used"] = "FAISS"
    
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def main():
    st.set_page_config("Chat PDF", layout="wide")
    st.markdown(get_html_template(), unsafe_allow_html=True)
    st.header("\U0001F4D6 Chat with PDF")
    st.write("Upload PDFs, ask questions, and get detailed answers.")

    # Display vector store status
    with st.sidebar:
        st.title("System Status")
        if pinecone_available:
            st.success("Pinecone: Connected")
        else:
            st.warning("Pinecone: Unavailable (using FAISS)")
        
        st.title("Upload and Process")
        pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    start_time = time.time()
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vector_store = get_hybrid_vector_store(text_chunks)
                    processing_time = time.time() - start_time
                    st.success(f"PDFs processed in {processing_time:.2f} seconds!")
                    st.info(f"Primary vector store: {vector_store['primary'].upper()}")
            else:
                st.warning("Please upload at least one PDF file.")

    # Chat interface
    if "conversation" not in st.session_state:
        st.session_state["conversation"] = []
    if "vector_store_used" not in st.session_state:
        st.session_state["vector_store_used"] = None

    user_question = st.text_input("Your Question:", placeholder="Type your question here...")

    if user_question:
        with st.spinner("Fetching answer..."):
            start_time = time.time()
            answer = user_input(user_question)
            query_time = time.time() - start_time
            st.session_state["conversation"].append(("user", user_question))
            st.session_state["conversation"].append(("bot", answer))
            st.session_state["conversation"].append(("system", f"Query processed using {st.session_state['vector_store_used']} in {query_time:.2f} seconds"))

    # Display conversation
    if st.session_state["conversation"]:
        for sender, message in st.session_state["conversation"]:
            if sender == "user":
                st.markdown(
                    f'<div class="chat-bubble user-bubble">\U0001F464 <span>{message}</span></div>',
                    unsafe_allow_html=True,
                )
            elif sender == "bot":
                st.markdown(
                    f'<div class="chat-bubble bot-bubble">\U0001F916 <span>{message}</span></div>',
                    unsafe_allow_html=True,
                )
            elif sender == "system":
                st.markdown(
                    f'<div class="chat-bubble system-bubble">\U0001F4BB <span>{message}</span></div>',
                    unsafe_allow_html=True,
                )

if __name__ == "__main__":
    main()
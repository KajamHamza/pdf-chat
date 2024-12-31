import streamlit as st
from htmlTemplate import get_html_template
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

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

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

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
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def main():
    st.set_page_config("Chat PDF", layout="wide")

    # Apply external HTML and CSS
    st.markdown(get_html_template(), unsafe_allow_html=True)

    st.header("\U0001F4D6 Chat with PDF")
    st.write("Upload PDFs, ask questions, and get detailed answers.")

    # Sidebar for file upload
    with st.sidebar:
        st.title("Upload and Process")
        pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("PDFs processed and indexed successfully!")
            else:
                st.warning("Please upload at least one PDF file.")

    # Chat-like interface
    if "conversation" not in st.session_state:
        st.session_state["conversation"] = []

    user_question = st.text_input("Your Question:", placeholder="Type your question here...")

    if user_question:
        with st.spinner("Fetching answer..."):
            answer = user_input(user_question)
            st.session_state["conversation"].append(("user", user_question))
            st.session_state["conversation"].append(("bot", answer))

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

if __name__ == "__main__":
    main()

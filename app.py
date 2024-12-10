import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS, Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from werkzeug.utils import secure_filename
import pinecone
from prompts import gen_prompt, acc_prompt, witty_prompt


def init_ses_states():
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "chat_pages" not in st.session_state:
        st.session_state.chat_pages = []


# Initialize Pinecone
def init_pinecone():
    pinecone.init(api_key="PINECONE_API_KEY", environment="us-west1-gcp")  # Replace with your actual Pinecone credentials
    index_name = "pdf-chat-index"
    try:
        pinecone.create_index(index_name, dimension=1536)  # Ensure this dimension matches your embeddings model
    except Exception as e:
        print("Index creation failed:", e)
    return pinecone.Index(index_name)


# Multiple PDFs
def get_pdfs_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        text += get_pdf_text(pdf)
    return text


# Single PDF
def get_pdf_text(pdf):
    text = ""
    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # Create FAISS vector store for local search
    faiss_vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    # Create Pinecone vector store for cloud storage
    pinecone_index = init_pinecone()
    pinecone_vectorstore = Pinecone.from_texts(text_chunks, embedding=embeddings, index_name=pinecone_index.index_name)
    
    return faiss_vectorstore, pinecone_vectorstore


def get_conversation_chain(faiss_vectorstore, pinecone_vectorstore, temp, model):
    llm = ChatOpenAI(temperature=temp, model_name=model)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    # Combine FAISS and Pinecone for hybrid retrieval
    retriever = faiss_vectorstore.as_retriever(search_type="hybrid", vectorstore=pinecone_vectorstore)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
    )
    return conversation_chain


def handle_userinput(user_question, prompt):
    response = st.session_state.conversation({'question': (prompt+user_question)})
    st.session_state.chat_history = response['chat_history']
    with st.spinner('Generating response...'):
        display_convo(prompt)
        

def display_convo(prompt):
    with st.container():
        for i, message in enumerate(reversed(st.session_state.chat_history)):
            if i % 2 == 0:
                st.markdown(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.markdown(user_template.replace("{{MSG}}", message.content[len(prompt):]), unsafe_allow_html=True)


def process_docs(pdf_docs, TEMP, MODEL):
    st.session_state["conversation"] = None
    st.session_state["chat_history"] = None
    st.session_state["user_question"] = ""

    raw_text = get_pdfs_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    
    faiss_vectorstore, pinecone_vectorstore = get_vectorstore(text_chunks)

    st.session_state.conversation = get_conversation_chain(faiss_vectorstore, pinecone_vectorstore, temp=TEMP, model=MODEL)
    st.session_state.pdf_processed = True


def set_prompt(PERSONALITY):
    if PERSONALITY=='general assistant': prompt = gen_prompt
    elif PERSONALITY == "academic": prompt = acc_prompt
    elif PERSONALITY == "witty": prompt = witty_prompt
    return prompt


def add_chat_page(title):
    st.session_state.chat_pages.append({"title":title})


def display_chat_page_titles():
    st.radio(label="Select Chat", options=[page["title"] for page in st.session_state.chat_pages], help="Choose a chat from the history.")


def main():
    st.set_page_config(page_title="Chat with PDFs", page_icon="📚", layout="wide")
    #st.write(css, unsafe_allow_html=True)
    init_ses_states()
    
    # Title and Subtitle
    st.title("📚 Chat with Your PDFs")
    st.markdown("Interact with your PDFs and ask questions about them! Powered by OpenAI, LangChain & Streamlit 🚀")
    
    # Sidebar: Settings
    with st.sidebar.expander("⚙️ Settings", expanded=True):
        MODEL = st.selectbox(label='🧠 Choose Model', options=['gpt-3.5-turbo', 'text-davinci-003', 'text-davinci-002', 'code-davinci-002'])
        PERSONALITY = st.selectbox(label='💬 Select Personality', options=['general assistant', 'academic', 'witty'])
        TEMP = st.slider("🌡️ Temperature", 0.0, 1.0, 0.5)
    
    # Sidebar: PDF Upload
    with st.sidebar.expander("📑 Upload Your Documents", expanded=True):
        pdf_docs = st.file_uploader("🔼 Upload PDFs", accept_multiple_files=True)
        if st.button("🚀 Process Files + New Chat"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    process_docs(pdf_docs, TEMP, MODEL)
            else: 
                st.caption("⚠️ Please upload at least one PDF.")
                st.session_state.pdf_processed = False

    if st.session_state.get("pdf_processed"):
        prompt = set_prompt(PERSONALITY)

        # User Input Form
        with st.form("user_input_form"):
            user_question = st.text_input("❓ Ask a question about your documents:")
            send_button = st.form_submit_button("✉️ Send")

        if send_button and user_question:
            if st.session_state.chat_history is None:
                title = user_question[:5]
                add_chat_page(title=title)
            handle_userinput(user_question, prompt)
    else: 
        st.caption("⚠️ Please upload at least one PDF to begin.")

if __name__ == '__main__':
    load_dotenv()
    main()

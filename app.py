import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# 1. Sidebar for file upload and instructions
st.sidebar.title("AI Chatbot")
pdf_file = st.sidebar.file_uploader("Upload a PDF document", type=["pdf"])
st.sidebar.info("Ask any question about your uploaded PDF!")

# 2. Load and process PDF only if uploaded
if pdf_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.read())

    # Load PDF and split into chunks
    loader = PyPDFLoader("temp.pdf")
    documents = loader.load_and_split()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(documents)

    # Create embeddings and FAISS vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.from_documents(texts, embeddings)
    retriever = db.as_retriever()

    # Load a small HuggingFace model with transformers pipeline
    hf_pipeline = pipeline(
        "text2text-generation", 
        model="google/flan-t5-small", 
        tokenizer="google/flan-t5-small"
    )
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # Create QA chain
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    # 3. Streamlit Chat Interface
    st.title("AI PDF Chatbot 🤖📄")

    # Initialize chat history in session_state
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # User question box
    user_question = st.text_input("Ask a question about the document:")

    if st.button("Ask") and user_question:
        response = qa.run(user_question)
        st.session_state.chat_history.append(("You", user_question))
        st.session_state.chat_history.append(("Bot", response))

    # Display conversation history
    for speaker, text in st.session_state.chat_history:
        if speaker == "You":
            st.markdown(f"**You:** {text}")
        else:
            st.markdown(f"**Bot:** {text}")

else:
    st.info("Please upload a PDF to get started.")

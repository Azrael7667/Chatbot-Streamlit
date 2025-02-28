import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from googletrans import Translator

# Load API Key from .env
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if api_key is None:
    st.error("GOOGLE_API_KEY is missing. Check your .env file.")
else:
    genai.configure(api_key=api_key)

translator = Translator()

# Function to translate text safely
def translate_text(text, dest_lang="ne"):
    if not text or text.strip() == "":
        return text  # Avoid translation errors on empty text
    try:
        translated = translator.translate(text, dest=dest_lang)
        if translated is None or translated.text is None:
            return text  # Return original text if translation fails
        return translated.text
    except Exception:
        return text  # Suppress translation errors by returning original text


# Function to extract text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Function to create a FAISS vector store and save it to session state
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    st.session_state["vector_store"] = vector_store  # Save to session state
    st.session_state["pdf_processed"] = True  # Mark PDF as processed
    return vector_store

# Function to create a conversational chain with FAISS retriever
def get_conversational_chain(vector_store):
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    retriever = vector_store.as_retriever()  # Use FAISS retriever

    chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    return chain

# Function to process user input
def user_input(user_question, output_language):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    try:
        translated_query = translate_text(user_question, dest_lang="ne")

        new_db = st.session_state.get("vector_store")  # Load stored FAISS vector store
        docs = new_db.similarity_search(translated_query)

        translated_docs = [translate_text(doc.page_content, dest_lang="en") for doc in docs]

        chain = get_conversational_chain(new_db)

        response = chain.invoke({"query": user_question})  # FIX: Correct key

        final_response = response["result"]

        if output_language == "Nepali":
            final_response = translate_text(final_response, dest_lang="ne")

        st.write("### Reply:")
        st.write(final_response)

    except Exception as e:
        st.error(f"Error in processing query: {e}")

# Main function for Streamlit UI
def main():
    st.set_page_config(page_title="Guff Space")
    st.header("Chat with PDF using Gemini")

    # Language Selection
    output_language = st.radio("Select Response Language", ["English", "Nepali"])

    # Sidebar for PDF Upload
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)

        # Ensure session state exists
        if "vector_store" not in st.session_state:
            st.session_state["vector_store"] = None
            st.session_state["pdf_processed"] = False

        # "Process PDF" Button
        if st.button("Process PDF", key="process_pdf"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text.strip():
                        st.error("No text could be extracted from the PDF. Try another file.")
                    else:
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)  # Save FAISS in session state
                        st.success("PDF Processing Done!")
            else:
                st.warning("Please upload at least one PDF file.")

    # User Query Section
    st.subheader("Ask a Question")
    user_question = st.text_input("Enter your question about the PDF:")

    # "Submit Query" Button
    if st.button("Submit Query", key="submit_query"):
        if st.session_state["pdf_processed"]:  # Ensure PDF is processed first
            user_input(user_question, output_language)
        else:
            st.warning("Please process the PDF before submitting a query.")

if __name__ == "__main__":
    main()

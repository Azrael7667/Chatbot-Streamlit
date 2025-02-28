# Chatbot Streamlit - PDF-Based Question Answering System

## Overview

This project is a Streamlit-based chatbot that allows users to upload PDFs and ask questions about the content. The chatbot processes the uploaded document, extracts text, converts it into embeddings using a sentence transformer model, and retrieves relevant information using FAISS (Facebook AI Similarity Search). The responses are generated using Google's Gemini AI model.

## Features

- Upload multiple PDF files
- Extract and process text from PDFs
- Convert text into embeddings using `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
- Store embeddings in FAISS for efficient retrieval
- Translate input queries and responses between English and Nepali
- Use Google Gemini AI for generating responses
- Interactive Streamlit interface for ease of use

## Technologies Used

- **Python**: Main programming language
- **Streamlit**: Web framework for interactive UI
- **PyPDF2**: PDF text extraction
- **FAISS**: Vector database for efficient similarity search
- **Sentence Transformers**: Embeddings model for text representation
- **Google Gemini AI**: LLM for response generation
- **Google Translate API**: For query and response translation
- **LangChain**: Framework for LLM-based applications

## Installation and Setup

### Prerequisites

Ensure that the following dependencies are installed:

- Python 3.8+
- Virtual environment (optional but recommended)
- Git for version control

### Clone the Repository

```sh
git clone https://github.com/Azrael7667/Chatbot-Streamlit.git
cd Chatbot-Streamlit
```

### Set Up a Virtual Environment (Recommended)

```sh
python -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate  # On Windows
```

### Install Dependencies

```sh
pip install -r requirements.txt
```

### Set Up API Keys

Create a `.env` file in the root directory and add the following:

```
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
```

### Run the Application

```sh
streamlit run app.py
```

## Usage

1. Upload one or more PDF files using the file uploader in the Streamlit sidebar.
2. Click the "Process PDF" button to extract and index the text.
3. Enter a question in the text box.
4. Click "Submit Query" to get an AI-generated response based on the PDF content.
5. Select the response language (English/Nepali) if translation is needed.

## Troubleshooting

- **Large File Issues:** Ensure that the `.gitignore` file is properly set up to exclude large files like `venv/` and model checkpoints.
- **Memory Issues:** If you experience memory-related errors, increase your system's swap space or reduce the batch size for embeddings.
- **API Key Issues:** Ensure that the `.env` file is correctly set up and contains valid API keys.

## Acknowledgments

- Sentence Transformers by Hugging Face
- FAISS by Facebook AI
- Google Gemini AI
- Streamlit for interactive applications


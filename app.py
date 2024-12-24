from flask import Flask, request, jsonify
import warnings
import os
from werkzeug.utils import secure_filename
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from typing import List

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'csv'}
DATA_FILE = 'data.txt'
FAISS_INDEX_FOLDER = 'faiss_index'
GOOGLE_API_KEY = "AIzaSyD6-NqQPoBRTXwYyi23sVLLfvQD5klKdjg" 

# Ensure required directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FAISS_INDEX_FOLDER, exist_ok=True)

# Create data.txt if it doesn't exist
if not os.path.exists(DATA_FILE):
    with open(DATA_FILE, 'w', encoding='utf-8') as f:
        f.write("")

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize LLM and embeddings
model = ChatGoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.2,
    convert_system_message_to_human=True
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

# Custom prompt template
QA_TEMPLATE = """Your task is to be an intelligent, concise, and polite assistant. 
Use the provided context to generate accurate responses to questions. 
If the answer isn't within the context, be honest and say you don't know. 
Always conclude with a warm and courteous note in a big heading.

Context: {context}
Question: {question}
Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(QA_TEMPLATE)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_file(file_path):
    """Process uploaded file and append to data.txt"""
    try:
        if file_path.endswith('.csv'):
            # Read CSV and convert to text
            df = pd.read_csv(file_path)
            text_content = df.to_string()
        else:
            # Read text file directly
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
        
        # Append to data.txt
        with open(DATA_FILE, 'a', encoding='utf-8') as f:
            f.write('\n' + text_content)
        return True
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return False
def update_vector_store():
    """Update the vector store with current data.txt content"""
    try:
        # Read the complete data
        with open(DATA_FILE, 'r', encoding='utf-8') as f:
            text_data = f.read()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_text(text_data)

        # Create vector store
        vector_index = FAISS.from_texts(texts, embeddings)

        # Save index and docstore separately
        vector_index.save_local(FAISS_INDEX_FOLDER)  # Removed the unsupported argument
        return vector_index
    except Exception as e:
        print(f"Error updating vector store: {str(e)}")
        raise


def get_qa_chain():
    """Initialize QA chain with current vector store"""
    try:
        # Load vector store with safe deserialization
        vector_index = FAISS.load_local(
            FAISS_INDEX_FOLDER,
            embeddings,
            allow_dangerous_deserialization=True 
        )

        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            model,
            retriever=vector_index.as_retriever(search_kwargs={"k": 5}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
        )
        return qa_chain
    except Exception as e:
        print(f"Error initializing QA chain: {str(e)}")
        raise


@app.route('/upload', methods=['POST'])
def upload_file():
    """Endpoint for file upload and processing"""
    try:
        # Ensure the request contains a file
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file part in the request',
                'debug_info': {
                    'content_type': request.content_type,
                    'files_received': list(request.files.keys())
                }
            }), 400

        file = request.files['file']

        # Check if a file was selected
        if file.filename == '':
            return jsonify({'error': 'No file selected for upload'}), 400

        # Validate the file type
        if not allowed_file(file.filename):
            return jsonify({
                'error': f'File type not allowed. Allowed types are: {ALLOWED_EXTENSIONS}',
                'received_filename': file.filename
            }), 400

        # Secure and save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Create the upload directory if it doesn't exist
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

        # Save the uploaded file
        file.save(filepath)

        # Process and update vector store
        if process_file(filepath):
            # Clean up the uploaded file
            os.remove(filepath)

            # Update vector store and respond with success
            update_vector_store()
            return jsonify({
                'message': 'File processed successfully',
                'filename': filename
            }), 200
        else:
            # Respond with an error if file processing fails
            return jsonify({'error': 'Failed to process the uploaded file'}), 500

    except Exception as e:
        # Log the error and send a structured error response
        print(f"Upload error: {str(e)}")  # Debugging
        return jsonify({
            'error': 'An unexpected error occurred while processing the upload',
            'details': str(e),
            'type': str(type(e))
        }), 500

import time
@app.route('/query', methods=['POST'])
def query():
    """Endpoint for querying the QA system"""
    try:
        data = request.json
        if not data or 'question' not in data:
            return jsonify({'error': 'No question provided'}), 400

        question = data['question']
        
        # Record start time
        start_time = time.time()
        
        # Get QA chain and process question
        qa_chain = get_qa_chain()
        result = qa_chain({'query': question})
        
        # Record end time
        end_time = time.time()
        
        # Calculate execution time
        execution_time = end_time - start_time
        
        # Extract and format source documents (if needed)
        sources = []
        for doc in result.get('source_documents', []):
            sources.append(doc.page_content[:200] + "...")  # First 200 chars of each source
        
        
        response = {
            'response': [
                f"Question: {question}",
                f"Execution Time: {execution_time:.2f} seconds",
                
                f"Answer: {result['result']}"
            ]
        }
        
        return jsonify(response), 200

    except Exception as e:
        print(f"Query error: {str(e)}")  # Debug print
        return jsonify({'error': str(e)}), 500


# Add a simple health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

if __name__ == '__main__':
    # Initialize with empty vector store if it doesn't exist
    if not os.path.exists(os.path.join(FAISS_INDEX_FOLDER, 'index.faiss')):
        update_vector_store()
    
    app.run(debug=True, host='0.0.0.0', port=5000)
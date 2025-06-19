# create_vectors.py
import sys
import json
import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()


# Configuración
PERSIST_DIR = os.getenv('PERSIST_DIR', './chroma_db')
METADATA_FILE = os.path.join(PERSIST_DIR, "file_metadata.json")
DOCUMENTS_DIR = "documentos"

def obtener_pdfs():
    return [os.path.join(DOCUMENTS_DIR, f) for f in os.listdir(DOCUMENTS_DIR) 
            if f.lower().endswith(".pdf")]

def cargar_metadata():
    if os.path.exists(METADATA_FILE):
        with open(METADATA_FILE, 'r') as f:
            return json.load(f)
    return {}

def guardar_metadata(metadata):
    os.makedirs(PERSIST_DIR, exist_ok=True)
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f)

def procesar_documentos():
    embeddings = OllamaEmbeddings(model=os.getenv('OLLAMA_EMBED_MODEL', 'nomic-embed-text'))
    metadata = cargar_metadata()
    pdf_files = obtener_pdfs()
    nuevos_archivos = []
    
    # Detectar cambios
    for file_path in pdf_files:
        mtime = os.path.getmtime(file_path)
        if metadata.get(file_path) != mtime:
            nuevos_archivos.append(file_path)
            metadata[file_path] = mtime
    
    # Procesar solo si hay cambios
    if nuevos_archivos:
        paginas = []
        for file_path in nuevos_archivos:
            loader = PyPDFLoader(file_path)
            paginas.extend(loader.load())
        
        text_splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=1000)
        docs = text_splitter.split_documents(paginas)
        
        if os.path.exists(PERSIST_DIR):
            vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
            vectorstore.add_documents(docs)
        else:
            vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=embeddings,
                persist_directory=PERSIST_DIR
            )
        
        guardar_metadata(metadata)
        print(f'Procesados {len(nuevos_archivos)} nuevos documentos')
    else:
        print('No hay documentos nuevos para procesar')


def crear_retriever():
    embeddings = OllamaEmbeddings(model=os.getenv('OLLAMA_EMBED_MODEL', 'nomic-embed-text'))
    
    if not os.path.exists(PERSIST_DIR):
        raise ValueError("Primero debes ejecutar CargaRag.py para crear la base de datos vectorial")
    
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )
    
    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={'score_threshold': 0.55}
    )


if __name__ == "__main__":
    procesar_documentos()
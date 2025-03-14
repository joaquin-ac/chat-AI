import sys
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

#LOGICA CARGAR DOCUMENTO PARA RAG
def obtener_pdfs():
    carpeta = "documentos"
    return [os.path.join(carpeta, f) for f in os.listdir(carpeta) if f.lower().endswith(".pdf")]

pdf_files = obtener_pdfs()

print(pdf_files)
#pdf_files = ["T2-2.1 Introduccion y DNS.pdf"]
#txt_files = ['Inscripción a Carreras de Grado - UNNE.txt']
paginas = []

#loader = WebBaseLoader(
#    ["https://www.unne.edu.ar/", "https://www.unne.edu.ar/nosotros/historia/"]
#)
#paginas.extend(loader.load())

for pdf_file in pdf_files:

    loader = PyPDFLoader(pdf_file)
    paginas.extend(loader.load())
    
#for txt_file in txt_files:
 #   loader = TextLoader(txt_file, encoding="utf-8")
 #   paginas.extend(loader.load())
# separar el texto de cada pdf en pedazos
text_spliter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)



docs = text_spliter.split_documents(paginas)
embeddings = OllamaEmbeddings(model="bge-m3")
vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={'k': 2})
print('Cargado')
#LOGICA CARGAR DOCUMENTO PARA RAG
#se extrae el vectorstore 
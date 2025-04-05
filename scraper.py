import asyncio
import aiohttp
from bs4 import BeautifulSoup
from io import BytesIO
import re
from pdfminer.high_level import extract_text
from typing import List, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords
import tiktoken

nltk.download('stopwords')
spanish_stop_words = stopwords.words('spanish')

def process_content(content: bytes, content_type: str, source: str) -> str:
    if 'pdf' in content_type or content.startswith(b'%PDF'):
        texto_extraido = [f'\nFuente: {source}\n', extract_text(BytesIO(content)), f'\nFuente: {source}\n']
        return '\n'.join(line for line in texto_extraido if line.strip())
    try:
        soup = BeautifulSoup(content.decode('utf-8', errors='ignore'), 'html.parser')
    except Exception as e:
        print(f"Error parsing HTML: {e}")
        return ''

    for tag in soup(['script', 'style', 'head', 'form', 'footer', 'header', 'aside']):
        tag.decompose()
    soup = decompose_non_p_links(soup)
    texto_extraido = [f'\nFuente: {source}\n', soup.get_text(separator='\n', strip=True), f'\nFuente: {source}\n']
    return '\n'.join(line for line in texto_extraido if line.strip())


async def fetch_url(url: str, session: aiohttp.ClientSession) -> str:
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept-Language': 'es-ES,es;q=0.9'
    }
    try:
        async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=200)) as res:
            content = await res.read()
            content_type = res.headers.get('Content-Type', '').lower()
            return await asyncio.to_thread(process_content, content, content_type, url)            
    except Exception as e:
        print(f"Error fetching URL: {str(e)}")
        return ''
    

async def scrape_web(urls: List[str]) -> List[str]:
    """devuelve una lista de contenidos obtenidos"""
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(url, session) for url in urls]
        results = await asyncio.gather(*tasks)
        return [result for result in results if result]


def decompose_non_p_links(soup):
    "devuelve el contenido descartando los links que no estan contenidos en etiquetas p generando ruido, se asume que son links a otro contenido informativo no relacionado"
    email_pattern = r'^[a-zA-Z0-9_. +-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    
    for a in soup.find_all('a'):
        if a.decomposed:
            continue
        
        if re.search(email_pattern, a.get_text().strip()):
            continue        
        
        parent_p_or_div = a.find_parent(['p', 'span', 'div'])
        
        if parent_p_or_div and parent_p_or_div.name == 'div':
            div_ancestor = parent_p_or_div
            current_element = a
            
            while current_element.parent != div_ancestor and current_element.parent is not None:
                current_element = current_element.parent
            
            current_element.decompose()
            
    return soup


# Función para contar tokens (split por espacios)
def count_tokens(text: str) -> int:
    return len(text.split())


def extract_relevant_text(query: str, documents: List[str], n: int) -> Union[str, List[str]]:
    if not documents:
        return "No se proporcionaron documentos."
    
    
    # Tokenizador preciso (ejemplo para GPT-3.5)
    enc = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(text: str) -> int:
        return len(enc.encode(text))  # Tokenización real
    
    selected_docs = []
    total_tokens = 0
    
    # Siempre agregar el documento más relevante (aunque supere n)
    if documents:
        primer_doc = documents[0]
        primer_doc_tokens = count_tokens(primer_doc)
        
        # Forzar inclusión incluso si supera n
        selected_docs.append(primer_doc)
        total_tokens = primer_doc_tokens
        
        # Iterar sobre los documentos restantes
        for doc in documents[1:]:
            print(total_tokens)
            current_tokens = count_tokens(doc)
            if total_tokens + current_tokens <= n:
                selected_docs.append(doc)
                total_tokens += current_tokens
    
    # Invertir el orden para que el más relevante quede al final
    return selected_docs[::-1] if selected_docs else []


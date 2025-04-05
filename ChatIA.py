import sys
import os
from typing import AsyncGenerator
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from SessionManager import SessionManager
from langchain_community.document_transformers import LongContextReorder
from QueryRefiner import query_refiner_chain, llm
from crawler import crawl_web
from scraper import scrape_web, extract_relevant_text
from datetime import datetime, timedelta, timezone
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv

# --- Configuración inicial ---
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables
load_dotenv()

# Configuración compartida
PERSIST_DIR = os.getenv('PERSIST_DIR', './chroma_db')

def crear_retriever():
    embeddings = OllamaEmbeddings(model=os.getenv('OLLAMA_EMBED_MODEL', 'nomic-embed-text'))
    
    if not os.path.exists(PERSIST_DIR):
        raise ValueError("Primero debes ejecutar create_vectors.py para crear la base de datos vectorial")
    
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )
    
    return vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={'score_threshold': 0.45}
    )


retriever = crear_retriever()

# Instancia para el manejo de sesiones
session_manager = SessionManager()

# --- Prompt principal para la respuesta del chatbot ---
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                """
                Proporciona respuestas en español utilizando el historial de conversación y la información recuperada, asegurando precisión, coherencia y contextualización. 
                Prioriza datos verificables de la UNNE (Universidad Nacional del Nordeste) en Argentina. 
                Incluye enlaces y correos electrónicos relevantes cuando estén disponibles.   

                ### **Directrices Principales**  
                1. **Contexto UNNE**: Asume que todas las consultas están relacionadas con la UNNE y su ámbito de influencia.  
                2. **Veracidad**: Nunca inventes información ni intentes predecirla o dar una idea general basada en otro conocimiento. Si no hay datos disponibles, indica claramente este hecho.  
                3. **Estructura**: Responde de forma clara, organizada y concisa. Vincula lógicamente la respuesta con el historial de conversación. 

                ### **Pasos de Ejecución**  
                1. **Analizar contexto**:  
                - Revisa el historial de conversación para captar la evolución y necesidades del usuario.  
                2. **Evaluar consulta**:  
                - Identifica si la pregunta requiere datos específicos de la UNNE (ej: trámites, contactos, normativas).  
                3. **Validar información**:  
                - Usa **exclusivamente** los datos recuperados del contexto. Nunca inventes información ni intentes predecirla o dar una idea general basada en otro conocimiento. Si no existen o no se menciona en el texto nada relevante, informa: *"No estoy interiorizado con esa información"* o *"No tengo suficiente conocimiento sobre [tema específico]. Te sugiero consultar [recurso oficial].*  
                4. **Construir respuesta**:  
                - Organiza la información en bloques temáticos con enlaces/correos integrados naturalmente.  
                - Ejemplo: *"Para inscripciones, completa el formulario en [Portal Académico](https://unne.edu.ar/inscripciones). Consultas: academica@unne.edu.ar".*  

                ---

                ### **Formato de Respuesta**  
                - **Lenguaje**: Formal pero accesible, evitando tecnicismos innecesarios.  
                - **Elementos clave**:  
                    Enlaces oficiales y correos verificados.  
                    Referencias a normativas/fechas si aplican.  
                    Evita suposiciones no respaldadas por la informacion disponible.  

                ---

                ### **Ejemplos**  

                **Caso 1 - Información Disponible**  
                - *Usuario*: "¿Cómo renovar la beca de investigación?"  
                - *Respuesta*: "La renovación de becas se gestiona mediante el [Sistema de Gestión de Becas](https://becas.unne.edu.ar). Requisitos: informe de avance y certificado de alumno regular. Plazo: hasta el 30/11. Dudas: becas@unne.edu.ar".  

                **Caso 2 - Información No Disponible**  
                - *Usuario*: "¿Cuál es el horario de la biblioteca en Resistencia?"  
                - *Respuesta*: "No tengo datos actualizados. Te recomiendo consultar el [sitio de Bibliotecas UNNE](https://biblioteca.unne.edu.ar) o contactarlos al bibcentral@unne.edu.ar".  

                **Caso 3 - 
                - *Usuario*: "¿Cómo renovar la beca de investigación?"  
                - *Respuesta* (si no hay datos en informacion disponible):  
                "No estoy interiorizado sobre el proceso de renovación de becas. Para obtener detalles precisos, visita el [Sistema de Gestión de Becas UNNE](https://becas.unne.edu.ar) o escribe a becas@unne.edu.ar".  
                ---  

                ### **Notas Finales**   
                - **Evitar referencias al texto**: Ten en cuenta que el usuario no conoce la 'informacion proporcionada', por lo tanto nunca menciones "el texto proporcionado" o "la información recuperada". Usa frases genéricas como:  
                    - *"No cuento con datos actualizados sobre..."*  
                    - *"Mi conocimiento sobre [área] es limitado. Te orientaría revisar..."*  
                - **Cortesía proactiva**: Ante la falta de información, ofrece siempre una alternativa de contacto o enlace oficial.
                - **Relevancia de Fechas:
                    Fecha actual: Menciónala explícitamente al responder si afecta plazos o procesos (ej: "Al día de hoy [fecha actual], el plazo está vigente hasta...").
                    Priorización: Resalta fechas críticas (inscripciones, renovaciones, eventos)  

                informacion disponible:
                
                {context}   
                """
            )
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "User Query: {input}\n\nResponse: ")
    ]
)
parser = StrOutputParser()
# Conecta el modelo al prompt principal
principal_chain = prompt_template | llm | parser

# --- Función para combinar documentos recuperados ---
def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# --- Función principal de crawling y scraping para streaming ---
async def crawl_and_scrape_chain_stream(question: str, session_id: str) -> AsyncGenerator[str, None]:
    # Recupera el historial de conversación de la sesión (suponiendo que session_manager ya está definido)
    chat_history = session_manager.get_history(session_id)
    refined_question = query_refiner_chain.invoke({
        "input": question,
        "chat_history": chat_history
    })
    
    print("Pregunta refinada:", refined_question)
    
    formatted_context = ctx_retriever(refined_question)

    #DESCOMENTAR PARA BUSCAR EN WEB   
    #if not formatted_context:   

        #formatted_context = await ctx_webSearch(refined_question)
    
    # Obtener fecha actual en formato DD/MM/AAAA
    utc_now = datetime.now(timezone.utc)
    argentina_time = utc_now - timedelta(hours=3)
    current_date = argentina_time.strftime("%d/%m/%Y")
    formatted_context.insert(-1, f'\n\nFecha Actual(d/m/y): {current_date}\n')
    
    print('Contexto final formateado:', formatted_context)
    
    full_response = ""
    # Genera la respuesta en streaming usando el pipeline del LLM (principal_chain debe ser compatible con async)
    async for chunk in principal_chain.astream({
        "input": question,
        "context": formatted_context,
        "chat_history": chat_history
    }):
        full_response += chunk
        yield chunk
    
    session_manager.update_history(
        session_id,
        HumanMessage(content=question),
        AIMessage(content=full_response)
    )




def ctx_retriever(refined_question):
    
    retrieved_information = retriever.invoke(refined_question)
    formatted_context = []
    
    if retrieved_information:
        # Extraer el contenido de cada Documento
        docs_context = [doc.page_content for doc in retrieved_information]
        
        # Reordenar (si es necesario)
        reordering = LongContextReorder()
        docs = [type("Doc", (object,), {"page_content": content})() for content in docs_context]
        reordered_docs = reordering.transform_documents(docs)
        
        # Agregar los nuevos elementos al final de la lista existente
        if reordered_docs:
            # Extrae el contenido de los docs reordenados y agrégalo a la lista
            new_content = [doc.page_content for doc in reordered_docs]
            formatted_context = new_content  # <--- Agrega la lista

        print('\n\nContenido de retriever:', new_content)
    return formatted_context


async def ctx_webSearch(refined_question):
    # Realiza crawling asíncrono para obtener las URLs relevantes
    urls = await crawl_web(refined_question)
    print("URLs recuperadas:", urls)
    
    # Realiza scraping asíncrono para extraer el contenido textual de las URLs encontradas
    scraped_contents = await scrape_web(urls)
    print("Contenidos extraídos:", scraped_contents)
        
    formatted_context = []
    # Si se obtuvo contenido, se extrae el fragmento más relevante semánticamente respecto a la pregunta refinada.
    if scraped_contents:
        formatted_context = extract_relevant_text(refined_question, scraped_contents, 2000)
        
        print('contenido mas relevante web: ', formatted_context)
        
    return formatted_context
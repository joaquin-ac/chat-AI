import sys
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
import os

# --- Configuración inicial ---
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')

# Cargar variables de entorno
load_dotenv()

# --- Modelo para Query Refiner (sin streaming) ---
#llm_refiner = OllamaLLM(model="llama3.2", temperature=0)
llm_refiner = ChatOpenAI(
    openai_api_base="https://api.deepseek.com/v1",
    openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
    model_name="deepseek-chat",
    temperature=0,
    streaming=False 
)

# --- Query Refiner (Reformulador de Preguntas) ---
query_refiner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                """
                Como experto en SEO y contexto UNNE, optimiza consultas usando historial de conversación. Reglas:  

                1. **Analizar historial**: Identifica términos clave previos (ej: si mencionó "becas", priorizarlo en nuevas consultas).  
                2. **Extraer palabras clave**: Verbo + sustantivos + contexto UNNE implícito (carreras, sedes, trámites).  
                3. **Inyectar contexto crítico**: Añade ubicación, tipo de trámite o área académica si el historial lo sugiere.  
                4. **Estructura natural**: Máximo 8 palabras, evitando redundancias, artículos, signos de puntuacion o guiones.  

                ---  

                ### **Ejemplos con Historial**  

                **Caso 1**  
                - *Historial*: "¿Dónde está la sede de ingeniería mecánica?"  
                - *Nueva consulta*: "Cómo me anoto?"  
                - *Optimizado*: "ingenieria mecanica UNNE"  

                **Caso 2**  
                - *Historial*: "Quiero una beca de posgrado"  
                - *Nueva consulta*: "requisitos para aplicar"  
                - *Optimizado*: "beca posgrado UNNE"  

                **Caso 3**  
                - *Historial*: "Horarios biblioteca Corrientes"  
                - *Nueva consulta*: "¿Tienen acceso virtual?"  
                - *Optimizado*: "biblioteca virtual UNNE"  

                ---  

                ### **Instrucción Final**  
                Devuelve **solo la consulta optimizada**, integrando palabras del historial cuando refuercen el contexto. Ejemplos:  
                    *Historial relevante*: "Estoy en la carrera de Sistemas"
                    *Consulta original*: "¿Cómo contacto a un docente?"  
                    *Optimizado*: "contacto docentes carrera sistemas UNNE" 
                    
                    *Historial relevante*: "Necesito legalizar mi título"
                    *Consulta original*: "¿Se puede hacer online?"
                    *Optimizado*: "legalizar titulo online UNNE"
                    
                    *Historial relevante*: "Quiero información sobre la carrera de Abogacia"
                    *Consulta original*: "¿Cuál es el plan de estudios?"
                    *Optimizado*: "plan estudio carrera abogacia UNNE"
                    
                    *Historial relevante*: "plan de estudio de la licenciatura en sistemas"
                    *Consulta original*: "cuanto tiempo tengo para inscribirme a esa carrera?"
                    *Optimizado*: "plazo inscripcion carrera licenciatura Sistemas UNNE"
                    
                """    
            )
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "Consulta original: {input}\n\nConsulta optimizada: ")
    ]
)

# Cadena de refinamiento (sin streaming)
query_refiner_chain = query_refiner_prompt | llm_refiner | StrOutputParser()
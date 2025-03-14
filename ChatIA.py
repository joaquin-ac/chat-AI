import sys
import re
from typing import AsyncGenerator
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from SessionManager import SessionManager
from langchain_community.document_transformers import LongContextReorder
from CargaRag import retriever
from QueryRefiner import query_refiner_chain, llm

# --- Configuración inicial ---
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')

# Instancia para el manejo de sesiones
session_manager = SessionManager()

# --- Prompt principal para la respuesta del chatbot ---
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                """
                Retrieved Information:  


                {context}  


                End of Retrieved Information  


                Respond to user querys in Spanish using the retrieved information and conversation history to provide accurate, consistent and contextualized responses. 
                Be sure to include relevant links and emails when available and avoid generating incorrect or fabricated information.  

                # Steps  

                1. Analyze the conversation history to understand the context and evolution of the dialogue.  
                2. Review the user's query considering the information accumulated in the conversation.  
                3. Construct a response that connects logically with the previous conversation, ensuring coherence and relevance.  
                4. If the response requires specific data, use retrieved information or verifiable sources; if no data is available, inform the user instead of inventing information.  
                5. When there are relevant links or e-mails in the retrieved information, include them in a clear and useful way.  

                # Response Format  

                - Respond in Spanish with clear, relevant and well-structured information.  
                - If there are useful links or emails, include them in a natural way in the response.  
                - If the information is not available, state this without making it up.  

                # Examples  

                ## Input  

                - **Conversation History**:  

                User: "Estoy planeando un viaje a Europa el próximo mes."  

                AI: "¡Genial! ¿A qué países planeas ir?"   

                - **User Query**: "¿Qué ciudades recomiendas visitar en Francia?"  

                ## Output  

                - Response: "Dado que ya estás organizando un viaje a Europa, te sugiero visitar París, Lyon y Marsella. Cada una ofrece experiencias culturales únicas. Puedes encontrar más información en [este enlace](https://ejemplo.com/guiaviaje)."  


                ## Input  

                - **Conversation History**:  

                User: "¿Cómo contacto al servicio técnico de la empresa X?"  

                AI: "Puedes consultar su sitio web oficial para obtener detalles de contacto." 

                - **User Query**: "¿Tienes su correo?"  

                ## Output.  

                - Response: "Sí, el soporte técnico de la empresa X puede ser contactado a través del correo soporte@empresa.com o en su página oficial: [https://empresa.com/soporte](https://empresa.com/soporte)."  


                ## Input (When no information is available).  

                - **User Query**: "¿Cuál es el contacto del servicio técnico de la empresa Y?"  

                ## Output  

                - Response: "No tengo información sobre el contacto del servicio técnico de la empresa Y. Te recomiendo visitar su sitio web oficial o buscar en sus canales de comunicación oficiales."  


                # Notes  

                - Do not generate false information or make up links or emails.  
                - Make sure the answers are clear, relevant and based on verifiable information.  
                - In case of doubt or lack of information, tell the user how to find reliable sources.  
                """
                
                
                #"Eres un asistente que responde consultas de nuestros usuarios."
                #"Utiliza el contexto recuperado y el historial de conversación para responder a la consulta."
                #"Responde de forma breve y concisa en un maximo de 300 caracteres."
                #"Agrega enlaces y correos relevantes a la respuesta."
                #"Menciona las fuentes de la informacion de la respuesta que proporcionaste."
                #"Si no tienes suficiente informacion para responder a la consulta di simplemente que no lo sabes."
                #"Evita decir cosas como 'debido al contexto proporcionado', 'segun el contexto proporcionado', 'en este contexto', etc."
                #"<context>"
                #"{context}"
                #"</context>"
            )
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "User Query: {input}\n\nResponse: ")
    ]
)
parser = StrOutputParser()
# Conecta el modelo al prompt principal
chain = prompt_template | llm | parser

# --- Función para combinar documentos recuperados ---
def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- Función principal con RAG y Query Refiner para streaming ---
async def rag_chain_stream(question: str, session_id: str) -> AsyncGenerator[str, None]:
    chat_history = session_manager.get_history(session_id)
    raw_context = retriever.invoke(question)
    refined_question = query_refiner_chain.invoke({
        "input": question,
        "context": raw_context,
        "chat_history": chat_history
    })
    cleaned_question = clean_response(refined_question)
    retrieved_docs = retriever.invoke(cleaned_question)
    reordering = LongContextReorder()
    reordered_docs = reordering.transform_documents(retrieved_docs)
    formatted_context = combine_docs(reordered_docs)
    full_response = ""
    async for chunk in chain.astream({
        "input": question,
        "context": formatted_context,
        "chat_history": chat_history
    }):
        full_response += chunk
        yield chunk
        
    print('pregunta refinada: ' + refined_question )    
    print('contexto: ' +formatted_context)
    
    session_manager.update_history(
        session_id,
        HumanMessage(content=question),
        AIMessage(content=full_response)
    )

# Función para limpiar la respuesta
def clean_response(response):
    return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
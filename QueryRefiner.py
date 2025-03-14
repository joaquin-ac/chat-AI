import sys
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- Configuración inicial ---
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')

# Carga del modelo de Ollama
llm = OllamaLLM(model="qwen2.5", temperature=0)

# --- Query Refiner (Reformulador de Preguntas) ---
query_refiner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                """
                Retrieved Information:
                
                {context}
                
                End of Retrieved Information
                
                Enhance user queries by incorporating conversation history for effective Google Search usage. 
                Provide an improved user query based on the entire conversation history so that it aligns better with the context and intent of the previous dialogue. Ensure that the rewritten query is optimized for obtaining relevant results in a Google Search. 

                # Steps
                 
                1. Analyze the provided conversation history to understand the context and progression of the dialogue. 
                2. Review the user's query considering the accumulated information from the conversation. 
                3. Adjust and refine the user's query to connect logically from the past conversation, maintaining coherence and relevance. 
                4. Optimize the query to enhance its suitability for a Google Search by making it more specific, using keywords and phrases that are likely to yield precise search results. 
                5. Returns only the refined query, without prefixes such as 'Consulta refinada: ' or 'Usuario: ' or 'en resumen el resultado es: '

                # Output Format
                 
                - Present the enhanced query in a concise and search-ready format in spanish. 

                # Example
                 
                ## Input
                 
                - **Conversation History**: 'User: ¿Como cocino pasta?\nAssistant: Hierve agua, añade sal, añade la pasta y cuécela durante 10 minutos.' \n
                - **User Query**: '¿Qué salsa va bien?' 

                ## Output 
                
                - Refined Query: '¿Qué salsa queda bien con la pasta hervida?' 

                # Notes 

                - Assume the conversation is provided in a coherent format. 
                - Focus on keeping queries concise and directly relevant to the conversation history. 
                - Queries should be rephrased for search efficiency without losing the original intent. 
                - Encourage clarity and use of common search terms that could improve search results.
                - Returns only the refined query, without prefixes such as 'Consulta refinada: ' or 'Usuario: '.
                - avoid phrases like 'en conclusion la query final es: ' or 'en resumen el resultado es: '. 
                """    
                
                #"Tu tarea es reformular entradas para mejorar la recuperación de información, en una base de datos vectorial\n"
                #"Dado el historial de conversación, una consulta y parte de contexto, reformula la consulta para que sea más clara, específica y que pueda ser entendida sin el historial de conversacion\n"
                #"Ejemplos:\n"
                #"1. consulta de usuario 1: 'que es el regimen de incopatibilidades?'\n"
                #"   consulta reformulada 1: '¿Qué es el régimen de incopatibilidad?'\n\n"
                #"2. consulta de usuario 2: 'a quienes afecta principalmente'\n"
                #"   consulta reformulada 2: '¿A qué personas afecta principalmente el régimen de incopatibilidad?'\n\n"
                #"Reglas:\n"
                #"- No respondas la consulta de usuario, solo reformula la consulta.\n"
                #"- Mantén el mismo significado de la consulta de usuario y no intentes traducirla\n"
                #"- Devuelve solo la consulta refinada, sin prefijos como 'Consulta refinada:' o 'Usuario:'."
            )
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "User Query: {input}\n\nRefined Query: ")
    ]
)
# Crear la cadena de refinamiento de consulta
query_refiner_chain = query_refiner_prompt | llm

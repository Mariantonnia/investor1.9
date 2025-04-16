import streamlit as st
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from langchain import LLMChain, PromptTemplate
from langchain_groq import ChatGroq
import os
import re
import json
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()

# Configurar el modelo LLM
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
    model="gemma2-9b-it",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Preguntas generales al inversor
preguntas_inversor = [
    "¿Cuál es tu objetivo principal al invertir?",
    "¿Cuál es tu horizonte temporal de inversión?",
    "¿Tienes experiencia previa invirtiendo en activos de mayor riesgo como acciones, criptomonedas o fondos alternativos?",
    "¿Estás dispuesto a sacrificar parte de la rentabilidad potencial a cambio de un impacto social o ambiental positivo?",
    "¿Qué opinas sobre el cambio climático?"
]

noticias = [
    "Repsol, entre las 50 empresas que más responsabilidad histórica tienen en el calentamiento global",
    "Amancio Ortega crea un fondo de 100 millones de euros para los afectados de la dana",
    "Freshly Cosmetics despide a 52 empleados en Reus, el 18% de la plantilla",
    "Wall Street y los mercados globales caen ante la incertidumbre por la guerra comercial y el temor a una recesión",
    "El mercado de criptomonedas se desploma: Bitcoin cae a 80.000 dólares, las altcoins se hunden en medio de una frenética liquidación",
    "Granada retrasa seis meses el inicio de la Zona de Bajas Emisiones, previsto hasta ahora para abril",
    "McDonald's donará a la Fundación Ronald McDonald todas las ganancias por ventas del Big Mac del 6 de diciembre",
    "El Gobierno autoriza a altos cargos públicos a irse a Indra, Escribano, CEOE, Barceló, Iberdrola o Airbus",
    "Las aportaciones a los planes de pensiones caen 10.000 millones en los últimos cuatro años",
]

# Plantillas y cadenas LLM
plantilla_evaluacion = """
Evalúa si esta respuesta del usuario es suficientemente detallada para un análisis ESG. 
Respuesta del usuario: {respuesta}
Si la respuesta es vaga, devuelve "False". Si es sustancial, devuelve "True".
"""
prompt_evaluacion = PromptTemplate(template=plantilla_evaluacion, input_variables=["respuesta"])
cadena_evaluacion = LLMChain(llm=llm, prompt=prompt_evaluacion)

plantilla_reaccion = """
Reacción del inversor: {reaccion}
Genera ÚNICAMENTE una pregunta de seguimiento enfocada en profundizar en la opinión del inversor.
"""
prompt_reaccion = PromptTemplate(template=plantilla_reaccion, input_variables=["reaccion"])
cadena_reaccion = LLMChain(llm=llm, prompt=prompt_reaccion)

plantilla_perfil = """
Análisis de reacciones y respuestas generales: {analisis}
Genera un perfil detallado del inversor basado en sus reacciones, respuestas generales y su aversión al riesgo.
Asigna una puntuación de 0 a 100 para cada pilar ESG y para el riesgo.
Devuelve en formato: Ambiental: [p], Social: [p], Gobernanza: [p], Riesgo: [p]
"""
prompt_perfil = PromptTemplate(template=plantilla_perfil, input_variables=["analisis"])
cadena_perfil = LLMChain(llm=llm, prompt=prompt_perfil)

# Procesamiento de respuestas
def procesar_respuesta_valida(user_input):
    with st.chat_message("bot", avatar="🤖"):
        st.write("Gracias por tu respuesta. Avanzando a la siguiente noticia...")

    st.session_state.historial.append({"tipo": "bot", "contenido": "Gracias por tu respuesta. Avanzando a la siguiente noticia..."})
    st.session_state.reacciones.append(user_input)
    st.session_state.contador += 1
    st.session_state.mostrada_noticia = False
    st.session_state.esperando_ampliacion = False
    st.rerun()

# Inicialización de estados
if "historial" not in st.session_state:
    st.session_state.historial = []
    st.session_state.contador = 0
    st.session_state.reacciones = []
    st.session_state.respuestas_inversor = []
    st.session_state.pregunta_general_idx = 0
    st.session_state.mostrada_noticia = False
    st.session_state.esperando_ampliacion = False

st.title("Chatbot de Análisis de Sentimiento")

# Mostrar historial
for mensaje in st.session_state.historial:
    with st.chat_message(mensaje["tipo"]):
        st.write(mensaje["contenido"])

# Preguntas generales antes de noticias
if st.session_state.pregunta_general_idx < len(preguntas_inversor):
    pregunta_actual = preguntas_inversor[st.session_state.pregunta_general_idx]
    with st.chat_message("bot", avatar="🤖"):
        st.write(pregunta_actual)

    user_input = st.chat_input("Tu respuesta...")
    if user_input:
        st.session_state.historial.append({"tipo": "user", "contenido": user_input})
        st.session_state.respuestas_inversor.append(user_input)
        st.session_state.pregunta_general_idx += 1
        st.rerun()

# Noticias después de preguntas generales
elif st.session_state.contador < len(noticias):
    if not st.session_state.mostrada_noticia:
        noticia = noticias[st.session_state.contador]
        with st.chat_message("bot", avatar="🤖"):
            st.write(f"¿Qué opinas sobre esta noticia? {noticia}")
        st.session_state.historial.append({"tipo": "bot", "contenido": noticia})
        st.session_state.mostrada_noticia = True
        st.session_state.esperando_ampliacion = False

    user_input = st.chat_input("Escribe tu respuesta aquí...")
    if user_input:
        st.session_state.historial.append({"tipo": "user", "contenido": user_input})

        if not st.session_state.esperando_ampliacion:
            evaluacion = cadena_evaluacion.run(respuesta=user_input).strip().lower()

            if evaluacion == "false":
                st.session_state.esperando_ampliacion = True
                pregunta_ampliacion = cadena_reaccion.run(reaccion=user_input).strip()
                with st.chat_message("bot", avatar="🤖"):
                    st.write(pregunta_ampliacion)
                st.session_state.historial.append({"tipo": "bot", "contenido": pregunta_ampliacion})
            else:
                procesar_respuesta_valida(user_input)
        else:
            procesar_respuesta_valida(user_input)

# Perfil final tras responder todas las noticias
else:
    analisis_total = "\n".join(st.session_state.respuestas_inversor + st.session_state.reacciones)
    perfil = cadena_perfil.run(analisis=analisis_total)
    with st.chat_message("bot", avatar="🤖"):
        st.write(f"**Perfil del inversor:** {perfil}")
    st.session_state.historial.append({"tipo": "bot", "contenido": f"**Perfil del inversor:** {perfil}"})

    puntuaciones = {
        "Ambiental": int(re.search(r"Ambiental: (\d+)", perfil).group(1)),
        "Social": int(re.search(r"Social: (\d+)", perfil).group(1)),
        "Gobernanza": int(re.search(r"Gobernanza: (\d+)", perfil).group(1)),
        "Riesgo": int(re.search(r"Riesgo: (\d+)", perfil).group(1)),
    }

    fig, ax = plt.subplots()
    ax.bar(puntuaciones.keys(), puntuaciones.values())
    ax.set_ylabel("Puntuación (0-100)")
    ax.set_title("Perfil del Inversor")
    st.pyplot(fig)

    # Guardar en Google Sheets
    try:
        creds_json_str = st.secrets["gcp_service_account"]
        creds_json = json.loads(creds_json_str)
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_json, scope)
        client = gspread.authorize(creds)
        sheet = client.open('BBDD_RESPUESTAS').sheet1

        fila = st.session_state.respuestas_inversor + st.session_state.reacciones + list(puntuaciones.values())
        sheet.append_row(fila)
        st.success("Datos guardados exitosamente en Google Sheets")
    except Exception as e:
        st.error(f"Error al guardar datos: {str(e)}")

# Mantener el foco en el input
st.markdown("""
<script>
document.addEventListener('DOMContentLoaded', () => {
    const input = document.querySelector('.stChatInput textarea');
    if(input) input.focus();
});
</script>
""", unsafe_allow_html=True)

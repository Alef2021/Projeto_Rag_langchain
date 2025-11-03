import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_chroma.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

#load .env ele nos permite ler a API KEY do Gemini
load_dotenv()
# Caminho do banco de dados vetorizado
CAMINHO_DB_BASE = f"Primeiro_agente_IA/db"
# O modelo LLM (para gerar a resposta)
GEMINI_LLM_MODEL = "gemini-2.5-flash" 
# O modelo de Embeddings (para vetorização)
GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"

st.set_page_config(page_title="🔮 Mestre Pokémon", page_icon="⚡")
st.title("⚡Seu Guia RAG de Todos Pokémon 1 Geração⚡")

# templete do prompt para chat
prompt_templete="""
Você é um Mestre Pokémon, um especialista em criaturas, regiões, tipos, do universo Pokémon. Suas respostas devem ser sempre focadas em Pokémon. 
Responda a pergunta do usuário:
{pergunta}
com base nessas informações a baixo:
{base_conhecimento} 
"""

#Garante que esta funcao seja executada uma vez
@st.cache_resource
def carregar_base_conhecimento():
    # Cria variavel de vetorizacao
    funcao_embedding = GoogleGenerativeAIEmbeddings(model=GEMINI_EMBEDDING_MODEL)
    # Carrega o banco de dados vetorizado
    db = Chroma(persist_directory=CAMINHO_DB_BASE, embedding_function=funcao_embedding)
    return db, funcao_embedding

#armazena os valores retornados da funcao
db, funcao_embedding = carregar_base_conhecimento()

def verifica_similaridade(pergunta: str):
    # usa a funcao de similaridade para encontrar a resosta mais coerente
    # k calibra o contexto dos chunks
    resltados_pesquisa = db.similarity_search_with_relevance_scores(pergunta, k=5) 
    
    # condicional para valibração, onde a primeira resposta - que 60% nao seria uma resposta confiavel 
    if len(resltados_pesquisa) == 0 or resltados_pesquisa[0][1] < 0.6:
        # **MODIFICAÇÃO AQUI:** Retorna uma string de erro em vez de None
        mensagem_erro = "Nem mesmo os arquivos da Pokéagenda contêm essa informação com a confiança necessária."
        print(mensagem_erro)
        return mensagem_erro
    
    texto_resultados = []
    # extrai o conteudo dos chunks retornados pelo separador content=, como ele é uma tupla pegamos apenas o primeiro item
    for resultado in resltados_pesquisa:
        texto = resultado[0].page_content
        texto_resultados.append(texto)

    #Usamos como separador
    base_conhecimento = "\n\n ----- Separador -----\n\n".join(texto_resultados)

    #Converte para um formato de entendimento do modelo LLM
    prompt = ChatPromptTemplate.from_template(prompt_templete)

    # Recupera as informacoes da base Chroma com as chaves do dict
    prompt_final = prompt.invoke({"pergunta":pergunta,"base_conhecimento":base_conhecimento})

    # cria o modelo para conversar com chat
    modelo = ChatGoogleGenerativeAI(model = GEMINI_LLM_MODEL)

    #Envia o prompt para modelo e armazena a resposta na variavel de retorno
    texto_resposta = modelo.invoke(prompt_final)
    return texto_resposta.content

#Construindo logica do chatbot streamlit 

# Iniciar conversa
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        AIMessage(content="Olá! Sou seu guia da 1ª Geração. Pergunte sobre qualquer Pokémon de Kanto!")
    ]

# Exibe o historico
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Entrada do Usuário
if user_input := st.chat_input("Pergunte sobre os Pokemons"):
    # Adiciona a pergunta do usuario ao historico
    st.session_state.messages.append(HumanMessage(content=user_input))
    
    # Exibe a pergunta do usuario
    with st.chat_message("user"):
        st.markdown(user_input)
    
    #Gera a resposta RAG
    with st.chat_message("assistant"):
        with st.spinner("Buscando na Pokéagenda de Alef..."):

            resposta_ia = verifica_similaridade(user_input) 
        
        st.markdown(resposta_ia)
        
        #Adiciona a resposta da IA ao histórico
        st.session_state.messages.append(AIMessage(content=resposta_ia))
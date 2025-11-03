from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

#load .env ele nos permite ler a API KEY do Gemini
load_dotenv()

#local onde a base de documentos pdf está salva
PASTA_BASE = f"Primeiro_agente_IA/base"
# O modelo de Embeddings (para vetorizar)
GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"


def criar_db(): 
    # Funcao que cria o Banco de dados vetorizado
    documentos = carregar_documentos()  # Usamos para carregar os arquivo PDF como base de conhecimento
    chunks = dividir_chunks(documentos) #dividir os documentos em pequenos textos(chunks)
    vetorizar_chunks(chunks) # vetorizar os chunks com processo embidding
    
    

def carregar_documentos():
    #Carregar o PDF
    carga = PyPDFDirectoryLoader(PASTA_BASE, glob = "*.pdf")
    #efetua uma um carga de listas
    documentos = carga.load()
    return documentos

def dividir_chunks(documentos):
    #Separa em pequenos textos que pode ser chamado de (chunks)
    separar_documentos = RecursiveCharacterTextSplitter(
        chunk_size = 4000, # seria como o intervalo de separacao entre os chunks
        chunk_overlap = 2000, # Uma margem de seguraça, sempre olha os ultimos 500 caracteres para ter coerencia sobre a quebra chunks
        length_function = len, # funcao len para contar os caracteres
        add_start_index = True # index 
    )
    #Separa os textos aplicando a logica explicada a cima
    chunks = separar_documentos.split_documents(documentos)
    return chunks


def vetorizar_chunks(chunks):
    # Cria variavel de vetorizacao
    embeddings = GoogleGenerativeAIEmbeddings(model=GEMINI_EMBEDDING_MODEL) 
    #cria o banco de dados vetorizado
    db = Chroma.from_documents(chunks, embeddings, persist_directory="Primeiro_agente_IA/db")
    print("DB Criado com sucesso!")

#chama função
criar_db()
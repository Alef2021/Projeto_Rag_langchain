from langchain_chroma.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

#load .env ele nos permite ler a API KEY do Gemini
load_dotenv()
# Caminho do banco de dados vetorizado
CAMINHO_DB_BASE = f"db"
# O modelo LLM (para gerar a resposta), para programação PRO é mais indicado
GEMINI_LLM_MODEL = "gemini-2.5-pro" 
# O modelo de Embeddings (para vetorização)
GEMINI_EMBEDDING_MODEL = "models/text-embedding-004"

# templete do prompt para chat
prompt_templete="""
Responda a pergunta do usuário:
{pergunta}
com base nessas informações a baixo:
{base_conhecimento} 

"""


 
def perguntar():
    pergunta = input("Escreva sua pergunta:")
    # Cria variavel de vetorizacao
    funcao_embedding = GoogleGenerativeAIEmbeddings(model=GEMINI_EMBEDDING_MODEL)
    # Carrega o banco de dados vetorizado
    db = Chroma(persist_directory=CAMINHO_DB_BASE, embedding_function=funcao_embedding)

    # usa a funcao de similaridade para encontrar a resosta mais coerente
    resltados_pesquisa = db.similarity_search_with_relevance_scores(pergunta,k=5) #k calibra o contexto do chunks
    # condicional para valibração, onde - que 70% nao seria uma resposta confiavel 
    if len(resltados_pesquisa) == 0 or resltados_pesquisa[0][1]< 0.7:
        print("Não sei te dizer isso")
        return 
    texto_resultados = []
    # extrai o conteudo dos chunks retornados pelo separador content=, como ele é uma tupla pegamos apenas o primeiro item
    for resultado in resltados_pesquisa:
        texto = resultado[0].page_content
        texto_resultados.append(texto)

    #Usamos como separador
    base_conhecimento = "\n\n ----- Separador -----\n\n".join(texto_resultados)
    #Converte para um formato de entendimento do modelo LLM
    prompt = ChatPromptTemplate.from_template(prompt_templete)
    # Recupera as informacoes da base Chroma com as chaves  do dict
    prompt = prompt.invoke({"pergunta":pergunta,"base_conhecimento":base_conhecimento})
    # cria o modelo para conversar com hat
    modelo = ChatGoogleGenerativeAI(model = GEMINI_LLM_MODEL)
    #Envia o prompt para modelo e armazena a resposta na variavel de retorno
    texto_resposta = modelo.invoke(prompt)
    print("Resposta da IA:",texto_resposta.content)

perguntar()
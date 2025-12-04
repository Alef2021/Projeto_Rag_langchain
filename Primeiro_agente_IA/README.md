# ⚡ Mestre Pokémon — Guia RAG da 1ª Geração

🟢 **Acesse Agora:** [projetoraglangchain-pokedexxalef.streamlit.app/](https://projetoraglangchain-pokedexxalef.streamlit.app/)

Uma aplicação interativa desenvolvida com **Streamlit** e **LangChain**, que utiliza **IA generativa do Google Gemini** e **busca vetorial (RAG)** para responder perguntas sobre Pokémon da 1ª geração (Kanto).  
A ferramenta atua como um “Mestre Pokémon”, capaz de buscar informações contextuais em uma base vetorizada e responder de forma inteligente e temática.

---

## 🌐 Deploy Online

A aplicação está disponível publicamente no Streamlit Cloud:  
👉 **[Acesse aqui o Mestre Pokémon](https://projetoraglangchain-pokedexxalef.streamlit.app/)**  

Você pode interagir diretamente com o assistente sem precisar instalar nada localmente.

---

## 🧠 Visão Geral

Este projeto implementa uma arquitetura **RAG (Retrieval-Augmented Generation)** para combinar:
- **Busca vetorial com ChromaDB**, que armazena e recupera dados sobre Pokémon.
- **Modelo de linguagem (LLM)** do **Google Gemini**, que gera respostas contextuais.
- **Interface interativa** em **Streamlit**, permitindo conversas estilo chatbot.

---

## 🚀 Tecnologias Utilizadas

| Tecnologia | Descrição |
|-------------|------------|
| **Python 3.10+** | Linguagem principal |
| **Streamlit** | Interface web interativa |
| **LangChain** | Framework para agentes e RAG |
| **Chroma** | Banco de dados vetorial para busca semântica |
| **Google Generative AI (Gemini)** | Modelo LLM e embeddings |
| **dotenv** | Leitura da variável de ambiente com a API key |

---

## 📂 Estrutura do Projeto

```
📦 Projeto_Pokedex
 ┣ 📁 Primeiro_agente_IA/
 ┃ ┗ 📁 db/                # Base vetorizada (Chroma)
 ┣ 📄 pokedex.py           # Código principal da aplicação
 ┣ 📄 .env                 # Contém a API_KEY do Gemini
 ┗ 📄 requirements.txt     # Dependências do projeto
```

---

## ⚙️ Instalação e Configuração

1. **Clone o repositório**
   ```bash
   git clone https://github.com/Alef2021/Projeto_Rag_langchain.git
    ```

2. **Crie e ative um ambiente virtual**
   ```bash
   python -m venv venv
   source venv/bin/activate     # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. **Instale as dependências**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure sua chave de API do Google**
   Crie um arquivo `.env` na raiz do projeto:
   ```env
   GOOGLE_API_KEY="sua_chave_api_aqui"
   ```

5. **Inicie a aplicação**
   ```bash
   streamlit run pokedex.py
   ```

---

## 💬 Uso

Após executar, a aplicação abrirá no navegador com o título:

> ⚡ Seu Guia RAG de Todos Pokémon 1ª Geração ⚡

Você pode digitar perguntas como:
- “Qual é o tipo do Charizard?”
- “Quais Pokémon são eficazes contra Gyarados?”
- “Onde posso encontrar Pikachu em Kanto?”

A IA buscará na base vetorizada e responderá de forma contextual e temática como um verdadeiro **Mestre Pokémon**.

---

## 🧩 Lógica Principal

1. **Carregamento da Base Vetorial**
   - A função `carregar_base_conhecimento()` inicializa o ChromaDB com embeddings do modelo `models/text-embedding-004`.

2. **Busca Semântica**
   - A função `verifica_similaridade()` busca trechos relevantes (`k=5`) e filtra resultados com relevância acima de 0.6.

3. **Geração da Resposta**
   - Cria um *prompt* com o contexto recuperado e a pergunta do usuário.
   - Envia o prompt para o modelo `gemini-2.5-flash` via `ChatGoogleGenerativeAI`.

4. **Interface**
   - Utiliza `st.chat_message` para exibir o histórico de perguntas e respostas de forma semelhante a um chat.

---

## ⚠️ Observações

- É necessário ter a **base vetorizada** previamente gerada e armazenada no diretório `Primeiro_agente_IA/db`.
- O projeto depende da **API Gemini** do Google — verifique se sua conta possui acesso.
- As respostas são limitadas à **1ª geração** de Pokémon (Kanto).

---

## 🧑‍💻 Autor

**Alef (Mestre Pokémon Dev)**  
📧 2021alef@gmail.com  
💼 [LinkedIn](https://www.linkedin.com/in/asouza94/) | 🐙 [GitHub](https://github.com/Alef2021)



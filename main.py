## Bibliotecas
#faiss-cpu
#python-dotenv
#langchain
#langchain_huggingface
#langchain_community
#langchain_groq
#langchain_core

## Bibliotecas
import string
import secrets
import os

## Funcoes do langchain
from langchain_huggingface                      import HuggingFaceEmbeddings
from langchain_community.vectorstores           import FAISS
from langchain_groq                             import ChatGroq
from langchain_core.prompts                     import ChatPromptTemplate
from langchain_core.prompts                     import MessagesPlaceholder
from langchain.chains                           import create_history_aware_retriever
from langchain.chains                           import create_retrieval_chain
from langchain.chains.combine_documents         import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history                import BaseChatMessageHistory
from langchain_core.runnables.history           import RunnableWithMessageHistory

## Modelo de embedding
model_name = 'all-mpnet-base-v2'

## Carrega o embedding
embeddings = HuggingFaceEmbeddings(model_name=model_name)

## Diretorio com vectorstores
vectorstore_dir = "mpnet_vectorstore"

# Cria a variavel do vectorstore vazia
db = None

# Lista os vectorstores no diretorio
vectorstore_files = [f for f in os.listdir(vectorstore_dir)]

# Junta todos os vectorstores no diretorio
for filename in vectorstore_files:
    filepath = os.path.join(vectorstore_dir, filename)

    current_vectorstore = FAISS.load_local(filepath, embeddings = embeddings, allow_dangerous_deserialization = True)

    if db is None:
        
        db = current_vectorstore
    else:
        
        db.merge_from(current_vectorstore)


## Chat com o groq
chat = ChatGroq(temperature = 0, model = "llama3-70b-8192")

## Questao de contexto/prompt de contexto
contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history")      ,
        ("human", "{input}")                     ,
    ]
)

## prompt de resposta
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


## cadeia de respostas
question_answer_chain = create_stuff_documents_chain(chat, qa_prompt)

## Aware retriever
history_aware_retriever = create_history_aware_retriever(
    chat, db.as_retriever(), contextualize_q_prompt
)

## RAG
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

## Funcao que gera um id de sessao para o usuario
def generate_session_id(length=6):
    characters = string.ascii_letters + string.digits
    session_id = ''.join(secrets.choice(characters) for _ in range(length))
    return session_id

## Cria um id para o usuario
session_id = generate_session_id()

## guarda o chat
store = {}

## Funcao para acessar o historico do chat
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

## Funcao do chatbot para o streamlit
def ChatBot(input):
    conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
    )
    
    response = conversational_rag_chain.invoke(
        {"input": input}, config={"configurable": {"session_id": session_id}}
    )
    
    answer  = response["answer" ]
    context = response["context"]
    
    # Extracting the source metadata from the response context
    sources = []
    for doc in context:
        source = doc.metadata["source"]
        if source:
            # Split the source to get the part after '\\' and remove '.txt'
            parts = source.split("\\")
            if len(parts) > 1:
                clean_source = parts[-1].replace(".txt", "")
                sources.append(clean_source)
            
    
    return answer, sources
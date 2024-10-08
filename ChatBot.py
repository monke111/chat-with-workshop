import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders.text import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_cohere import CohereEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

def get_vectorstore():
    loader = TextLoader('./index.md')
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    index_name = "chatwithworkshop"
    vector_store = PineconeVectorStore.from_documents(document_chunks, CohereEmbeddings(model='embed-english-light-v3.0'), index_name=index_name)
    return vector_store

def get_context_retriever_chain(vector_store, groq_api_key):
    llm = ChatGroq(api_key=groq_api_key, model="llama3-8b-8192")
    retriever = vector_store.as_retriever()
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the current question. " +
         "Don't leave out any relevant keywords. Only return the query and no other text.",)
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain

def get_conversational_rag_chain(retriever_chain, groq_api_key):
    llm = ChatGroq(api_key=groq_api_key, model="llama3-8b-8192")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a personal tutor for students attending a workshop. You impersonate the workshop instructor. " +
         "Answer the user's questions based on the below context. " +
         "Whenever it makes sense, provide links to pages that contain more information about the topic from the given context. " +
         "Format your messages in markdown format.\n\n" +
         "Context:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(user_input, groq_api_key):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store, groq_api_key)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain, groq_api_key)
    response_stream = conversation_rag_chain.stream({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    for chunk in response_stream:
        content = chunk.get("answer", "")
        yield content

# App config
st.set_page_config(page_title="Chat with Workshop", page_icon="📕")

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

st.title("Chat with Workshop")
st.markdown(
    """
    Let us know what we can improve about the chatbot here: 
    [Feedback Form](https://forms.gle/tKQ4QMYBfe4jEgGq7)
"""
)
st.sidebar.header("Chat with Workshop")

# Get Groq API key from user
groq_api_key = st.text_input("Enter your Groq API key:", type="password")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello, I am your AI workshop Instructor. Ask me any doubts related to the workshop."),
    ]
if "vector_store" not in st.session_state:
    st.session_state.vector_store = get_vectorstore()    

# Display conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)

# User input
user_query = st.chat_input("Type your message here...")
if user_query is not None and user_query != "":
    if not groq_api_key:
        st.error("Please enter your Groq API key.")
    else:
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        with st.chat_message("Human"):
            st.markdown(user_query)
        
        with st.chat_message("AI"):
            response = st.write_stream(get_response(user_query, groq_api_key))
        st.session_state.chat_history.append(AIMessage(content=response))
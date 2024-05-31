import tempfile
import shutil
from operator import itemgetter
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

TEMPLATE = """You're TextBook-Assistant. You're an expert in analyzing history and economics textbooks. You should only rely on {context} and on the previous message history as context, and from that you build a context and history-aware to reply. 
MAKE SURE YOU MENTION THE NAME OF THE FILE ALONG WITH PAGE NUMBERS OF INFORMATION FROM THE METADATA AT THE END OF YOUR RESPONSE EVERYTIME IN THIS FORMAT [File Name : Page Number].
If you don't know the answer, just say that you don't know; don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
"""

def load_pdf_text(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        shutil.copyfileobj(uploaded_file, temp_file)
        temp_file_path = temp_file.name

    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()
    for doc in docs:
        doc.metadata['filename'] = uploaded_file.name
    total_text = "\n".join(doc.page_content for doc in docs)
    doc_length = len(total_text)

    return docs, doc_length

def determine_optimal_chunk_size(doc_length):
    if doc_length < 5000:
        chunk_size = 500
        chunk_overlap = 100
    elif doc_length < 20000:
        chunk_size = 1000
        chunk_overlap = 250
    else:
        chunk_size = 2000
        chunk_overlap = 500
    return chunk_size, chunk_overlap

def chunk_and_store_in_vector_store(docs, chunk_size, chunk_overlap, token, qurl, qapi):
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=token, model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(docs)

    vectorstore = Qdrant.from_documents(
        documents=splits, embedding=embeddings, url=qurl, api_key=qapi, collection_name='test1234'
    )
    return vectorstore

def process_user_input(user_query, vectorstore, token, chat_history,sessionid):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

    store = {}
    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]
    llm = HuggingFaceEndpoint(
        huggingfacehub_api_token=token,
        repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
        task="text-generation",
        max_new_tokens=512,
        top_k=50,
        top_p=0.8,
        temperature=0.1,
        repetition_penalty=1
    )

    prompt_template = ChatPromptTemplate.from_messages(
    [("system", TEMPLATE),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")]
)   
    context = itemgetter("question") | retriever | format_docs
    first_step = RunnablePassthrough.assign(context=context)
    chain = first_step | prompt_template | llm
    with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history=get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history")
    llm_response = with_message_history.invoke({"question":user_query},config={
        "configurable": {"session_id": sessionid}
    })
    final_output = llm_response
    return final_output

def format_docs(docs):
    formatted_docs = []
    for doc in docs:
        content = doc.page_content
        page = doc.metadata.get('page') + 1
        source = doc.metadata.get('filename')
        formatted_docs.append(f"{content} Source: {source} : {page}")
    return "\n\n".join(formatted_docs)
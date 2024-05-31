import tempfile
import shutil
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain.chains import StuffDocumentsChain, LLMChain

TEMPLATE = """You're TextBook-Assistant. You're an expert in analyzing history and economics textbooks.
Use the following pieces of context to answer the question at the end.
MAKE SURE YOU MENTION THE NAME OF THE FILE ALONG WITH PAGE NUMBERS OF INFORMATION FROM THE METADATA AT THE END OF YOUR RESPONSE EVERYTIME IN THIS FORMAT [File Name : Page Number].
If you don't know the answer, just say that you don't know; don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.

{context}

Question: {question}

Answer:"""

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
        documents=splits, embedding=embeddings, url=qurl, api_key=qapi, collection_name='HistoryTest'
    )
    return vectorstore

def process_user_input(user_query,usq, vectorstore, token, chat_history):
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    
    template2 = PromptTemplate.from_template(
    """Analyze the chat history and follow up question which might reference context in the chat history, If the question is direct which doesnt really refer to the chat history just return as it is , understand what is the user exactly asking, into  
    a standalone question which can be understood. Chat History: {chat_history}
    Follow up question: {question}
    YOUR FINAL OUTPUT SHOULD JUST BE THE STANDALONE QUESTION NOTHING ELSE."""
    )
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
    
    question_generator_chain = LLMChain(llm=llm, prompt=template2)
    generated_question = question_generator_chain.run({'question': user_query, 'chat_history': chat_history})
    original_string=generated_question
    if original_string.startswith("Standalone question: "):
        original_string = original_string[len("Standalone question: "):]
    index = original_string.find("Analyze")
    standalone_question = original_string[:index].strip() if index != -1 else original_string.strip()
    index = generated_question.find("Analyze")
    standalone_question = generated_question[:index].strip()
    qu=standalone_question+usq
    template = TEMPLATE
    custom_rag_prompt = PromptTemplate.from_template(template)
    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | custom_rag_prompt
        | llm
        | StrOutputParser()
    )
    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)
    llm_response = rag_chain_with_source.invoke(qu)
    final_output = llm_response['answer']
    return final_output

def format_docs(docs):
    formatted_docs = []
    for doc in docs:
        content = doc.page_content
        page = doc.metadata.get('page') + 1
        source = doc.metadata.get('filename')
        formatted_docs.append(f"{content} Source: {source} : {page}")
    return "\n\n".join(formatted_docs)

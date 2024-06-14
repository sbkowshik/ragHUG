
import streamlit as st
from rag_chain import load_doc_text, determine_optimal_chunk_size, chunk_and_store_in_vector_store, process_user_input
import uuid

token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
qurl = st.secrets["QDRANT_URL"]
qapi = st.secrets["QDRANT_API"]
uapi = st.secrets["UNST_API"]

def main():
    
    st.set_page_config(page_title="RAG Assistant")
    st.title("RAG Assistant")
    session_id = str(uuid.uuid4)
    if 'vectorstore' not in st.session_state:
        st.session_state['vectorstore'] = None

    if 'bmv' not in st.session_state:
        st.session_state['splits'] = None
        
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
    
    with st.sidebar:
        st.title("Kowshik S B")
        
        st.session_state['source_docs'] = st.file_uploader("Upload documents", accept_multiple_files=True)
        
        if st.button("Submit"):
            with st.spinner("Processing DOCs..."):
                if st.session_state['source_docs']:
                    for source_doc in st.session_state['source_docs']:
                        docs, doc_length = load_doc_text(source_doc,uapi)
                        chunk_size, chunk_overlap = determine_optimal_chunk_size(doc_length)
                        st.session_state['vectorstore'], st.session_state['bmv'] = chunk_and_store_in_vector_store(
                            docs, chunk_size, chunk_overlap, token=token, qapi=qapi, qurl=qurl
                        )
                    st.success("DOCs Processed")
                else:
                    st.info("Please upload documents")
    
    for message in st.session_state['messages']:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    user_query = st.chat_input("Ask a question")
    
    if user_query and st.session_state['vectorstore']:
        usq = ". \t You must mention the file name along with page numbers only if it exists of the relevant information only from the METADATA in this format [File Name : Page Number(Optional)]"
        with st.chat_message("user"):
            st.write(user_query)
        st.session_state['messages'].append({"role": "user", "content": user_query})
        ch = []
        for msg in st.session_state['messages']:
            ch.append((msg['role'], msg['content']))
        chat_history = ch
        with st.chat_message("assistant"):
            llm_answer = process_user_input(user_query, usq, st.session_state['vectorstore'], token, chat_history,st.session_state['splits'])
            st.write(llm_answer)
        st.session_state['messages'].append({"role": "assistant", "content": llm_answer})
    elif user_query:
        st.warning("Please upload PDFs")

if __name__ == "__main__":
    main()

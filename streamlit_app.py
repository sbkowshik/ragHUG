import streamlit as st
from rag_chain import load_pdf_text, determine_optimal_chunk_size, chunk_and_store_in_vector_store, process_user_input
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)

token=st.secrets["HUGGINGFACEHUB_API_TOKEN"]
qurl=st.secrets["QDRANT_URL"]
qapi=st.secrets["QDRANT_API"]

def main():
    st.set_page_config("Social Studies RAG Assistant")
    st.title("Social Studies RAG Assistant")
    if 'vectorstore' not in st.session_state:
        st.session_state['vectorstore'] = None
    with st.sidebar:
        st.title("Kowshik S B")
        
        st.session_state['source_docs'] = st.file_uploader(" ",accept_multiple_files=True)
        if st.button("Submit"):
            with st.spinner(" "):
                if  st.session_state['source_docs'] is not None:
                    for source_doc in  st.session_state['source_docs']:
                        docs, doc_length = load_pdf_text(source_doc)
                        chunk_size, chunk_overlap = determine_optimal_chunk_size(doc_length)
                        st.session_state['vectorstore'] = chunk_and_store_in_vector_store(docs, chunk_size, chunk_overlap,token=token,qapi=qapi,qurl=qurl)
                    st.success("PDFs Processed")
                else:
                    st.info("Upload PDF")
    st.session_state['chat_history']=[]
    for message in st.session_state['chat_history']:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_query = st.chat_input("Ask question")
    usq= "You must mention the file name along with page numbers of the relevant information only from the METADATA in this format [File Name : Page Numbers]"
    if user_query and st.session_state['vectorstore']:
        print(user_query)
        with st.chat_message("user",avatar="😺"):
            st.markdown(user_query)
        with st.chat_message("assistant",avatar="🦖"):
            llm_answer = process_user_input(user_query + usq, st.session_state['vectorstore'],token=token)
            st.markdown(llm_answer)
        st.session_state['chat_history'].append({"role": "assistant", "content": llm_answer})
    elif user_query:
        st.warning("Upload PDF")

if __name__ == "__main__":
    main()
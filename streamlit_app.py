import streamlit as st
from rag_chain import load_pdf_text, determine_optimal_chunk_size, chunk_and_store_in_vector_store, process_user_input
from langchain_community.callbacks.streamlit import (
    StreamlitCallbackHandler,
)

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
                        st.session_state['vectorstore'] = chunk_and_store_in_vector_store(docs, chunk_size, chunk_overlap)
                    st.success("PDFs Processed")
                else:
                    st.info("Upload PDF")

    user_query = st.chat_input("Ask question")
    usq= "You must tell me the file name along with page numbers of the relevant information in this format (File Name : Page Numbers)"
    if user_query and st.session_state['vectorstore']:
        print(user_query)
        with st.chat_message("user",avatar="😺"):
            st.markdown(user_query)
        with st.chat_message("assistant",avatar="🦖"):
            llm_answer = process_user_input(user_query + usq, st.session_state['vectorstore'])
            st.markdown(llm_answer)

    elif user_query:
        st.warning("Upload PDF")

if __name__ == "__main__":
    main()

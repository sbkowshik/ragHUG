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
        pdf_doc = st.file_uploader(" ",accept_multiple_files=False)
        if st.button("Submit"):
            with st.spinner(" "):
                if pdf_doc is not None:
                    docs, doc_length = load_pdf_text(pdf_doc)
                    chunk_size, chunk_overlap = determine_optimal_chunk_size(doc_length)
                    st.session_state['vectorstore'] = chunk_and_store_in_vector_store(docs, chunk_size, chunk_overlap,doc_length)
                    st.success("PDF Processed")
                else:
                    st.error("Upload PDF")

    user_query = st.chat_input("Ask question")
    usq=user_query + 'You must tell me the page numbers of the relevant information you got from the textbook'
    if user_query and st.session_state['vectorstore']:
        with st.chat_message("user",avatar="😺"):
            st.markdown(user_query)
        with st.chat_message("assistant",avatar="🦖"):
            llm_answer = process_user_input(usq, st.session_state['vectorstore'])
            st.markdown(llm_answer)

    elif user_query:
        st.warning("Upload PDF")

if __name__ == "__main__":
    main()


import streamlit as st
from rag_chain import load_pdf_text, determine_optimal_chunk_size, chunk_and_store_in_vector_store, process_user_input

token = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
qurl = st.secrets["QDRANT_URL"]
qapi = st.secrets["QDRANT_API"]

def main():
    st.set_page_config(page_title="Social Studies RAG Assistant")
    st.title("Social Studies RAG Assistant")

    if 'vectorstore' not in st.session_state:
        st.session_state['vectorstore'] = None
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []

    with st.sidebar:
        st.title("Kowshik S B")
        st.session_state['source_docs'] = st.file_uploader("Upload your PDFs", accept_multiple_files=True)
        
        if st.button("Submit"):
            if st.session_state['source_docs'] is not None:
                with st.spinner("Processing PDFs..."):
                    for source_doc in st.session_state['source_docs']:
                        docs, doc_length = load_pdf_text(source_doc)
                        chunk_size, chunk_overlap = determine_optimal_chunk_size(doc_length)
                        st.session_state['vectorstore'] = chunk_and_store_in_vector_store(
                            docs, chunk_size, chunk_overlap, token=token, qapi=qapi, qurl=qurl
                        )
                    st.success("PDFs Processed")
            else:
                st.info("Please upload at least one PDF.")

    user_query = st.chat_input("Ask a question")
    usq = "You must mention the file name along with page numbers of the relevant information only from the METADATA in this format [File Name : Page Numbers]"

    if user_query:
        if st.session_state['vectorstore']:
            with st.chat_message("user", avatar="😺"):
                st.markdown(user_query)
            
            with st.chat_message("assistant", avatar="🦖"):
                llm_answer = process_user_input(user_query + usq, st.session_state['vectorstore'], token, st.session_state['chat_history'])
                st.markdown(llm_answer)

                # Update chat history
                st.session_state['chat_history'].append({"user": user_query, "assistant": llm_answer})
        else:
            st.warning("Please upload and process a PDF first.")

if __name__ == "__main__":
    main()

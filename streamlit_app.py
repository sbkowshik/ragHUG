import streamlit as st
from rag_chain import load_pdf_text, determine_optimal_chunk_size, chunk_and_store_in_vector_store, process_user_input

def create_chat_bubble(text):
    chat_bubble_html = f"""
    <style>
    .chat-bubble {{
        max-width: 100%;
        margin: 10px;
        padding: 10px;
        background-color: #262730;
        border-radius: 16px;
        border: 1px solid #36454F;
    }}
    .chat-container {{
        display: flex;
        flex-direction: column;
        align-items: flex-start;
    }}
    </style>
    <div class="chat-container">
        <div class="chat-bubble">
            {text}
        </div>
    </div>
    """
    return chat_bubble_html

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
                    st.session_state['vectorstore'] = chunk_and_store_in_vector_store(docs, chunk_size, chunk_overlap)
                    st.success("PDF Processed")
                else:
                    st.error("Upload PDF")

    user_query = st.text_input("Ask question")
    usq=user_query + 'You should Mention the page number of the information you got from the textbook [Page No]'
    if user_query and st.session_state['vectorstore']:
        llm_answer = process_user_input(user_query, st.session_state['vectorstore'])
        st.markdown(create_chat_bubble(llm_answer), unsafe_allow_html=True)

    elif user_query:
        st.warning("Upload PDF")

if __name__ == "__main__":
    main()


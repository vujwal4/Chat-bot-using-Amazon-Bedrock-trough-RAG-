import streamlit as st
import backend as back

st.set_page_config(page_title = "Chat Bot for leave policy with RAG")

new_title = '<p> Chat Bot for leave policy with RAG</p>'
st.markdown(new_title , unsafe_allow_html = True)

if 'vector_insex' not in st.session_state:
    with st.spinner("Fetching Data"):
        st.session_state.vector_index = back.RAG_pdf()

input_text = st.text_area("Input text" , label_visibility = "hidden")
go_button = st.button("Your Question" , type = "primary")

if go_button:
    with st.spinner("Gtting th Answer"):
        response_content = back.RAG_response(vectorstore = st.session_state.vector_index ,question = input_text)
        st.write(response_content)

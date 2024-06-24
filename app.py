import streamlit as st
from PyPDF2 import PdfReader

st.title('Chatbot')

def main():
    st.header("Hello")


    # upload a pdf file


    pdf=st.file_uploader("Upload your pdf", type="pdf")

    if pdf is not None:
        pdf_reader=PdfReader(pdf)
        text=""
        for page in pdf_reader.pages:
            text+=page.extract_text()
        st.write(text)






if __name__=='__main__':
    main()
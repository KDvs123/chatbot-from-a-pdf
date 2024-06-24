import os
from langchain import InMemoryDocstore
import streamlit as st
import faiss
from dotenv import load_dotenv

from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS


st.title('Chatbot')



def main():
    st.header("Hello")
    load_dotenv()



    # upload a pdf file


    pdf=st.file_uploader("Upload your pdf", type="pdf")
   
    if pdf is not None:
        pdf_reader=PdfReader(pdf)
        text=""
        for page in pdf_reader.pages:
            text+=page.extract_text()
        
        text_splitter= RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

        chunks = text_splitter.split_text(text=text)

        #embeddings

        
        store_name = pdf.name[:-4]
        index_folder = f'src/faiss_store/{store_name}'

        if os.path.exists(index_folder):
            try:
                vectorstores = FAISS.load_local(index_folder, OpenAIEmbeddings(),allow_dangerous_deserialization=True)
                # st.write("Loaded vectorstores from local storage")
            except Exception as e:
                st.write(f"Failed to load local storage: {e}")
        else:
            try: 
                embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
                embedding_dim=1024
                faiss_index = faiss.IndexFlatL2(embedding_dim)
                # Initialize the docstore
                docstore = InMemoryDocstore()
                # Initialize the index_to_docstore_id
                index_to_docstore_id = {}
                # vectorstores = FAISS(chunks,embeddings)
                vectorstores = FAISS(embedding_function=embeddings, index=faiss_index, docstore=docstore, index_to_docstore_id=index_to_docstore_id)
                vectorstores.save_local(index_folder)
                # st.write("Saved vectorstores to local storage")
            except Exception as e:
                st.write(f"Failed to save local storage: {e}")

            
            # Accept user questions/ query

            query=st.text_input("Ask Questions about the PDF:")
            st.write(query)
            

        


        


        



    






if __name__=='__main__':
    main()
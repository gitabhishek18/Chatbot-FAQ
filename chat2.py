import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from sentence_transformers import SentenceTransformer
from langchain_community.llms import Ollama
llm = Ollama(model="llama3")
st.header('Lord chatbot :]')
with st.sidebar:
    st.title('drop your documents=')
    file=st.file_uploader('upload a pdf file and start asking questions',type='pdf')
if file is not None:
    pdf_file=PdfReader(file)
    doc=[]
    for i in pdf_file.pages:
        text = i.extract_text()  
        if text: 
            doc.append(Document(page_content=text))
    all_text = " ".join([doc1.page_content for doc1 in doc])
    text_spitter=RecursiveCharacterTextSplitter(
        separators='\n',
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks=text_spitter.split_text(all_text)


    model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


    

    vector_store=FAISS.from_texts(chunks,model)

    user_question=st.text_input('enter a question')

    if user_question:
        match=vector_store.similarity_search(user_question)
        chain=load_qa_chain(llm,chain_type="stuff")
        response=chain.run(input_documents=match,question=user_question)
        st.write(response)

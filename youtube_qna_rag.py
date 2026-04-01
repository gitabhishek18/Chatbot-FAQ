from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM,OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

import re
from urllib.parse import urlparse, parse_qs

import streamlit as st

def extract_video_id(url):
    if not url:
        return None
    match = re.search(r'(?:youtu\.be\/|youtube\.com\/(?:embed\/|v\/|watch\?v=|watch\?.*&v=))([\w-]{11})', url)   
    if match:
        return match.group(1)
    try:
        query = urlparse(url).query
        params = parse_qs(query)
        if 'v' in params:
            return params['v'][0]
    except Exception:
        pass
    return None
if 'transcript' not in st.session_state:
    st.session_state.transcript = ""
if 'video_id' not in st.session_state:
    st.session_state.video_id = ""
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
@st.cache_resource
def get_youtube_api():
    return YouTubeTranscriptApi()
st.set_page_config(page_title="Youtube (Q n A) bot",layout="wide")
st.title("Chat from youtube video .")

with st.sidebar:    
    st.header("Confirguration :")
    url=st.text_input("Enter your link here :",key="input_url")
    if st.button("Process video"):
        video_id=extract_video_id(url)
        if video_id:
            st.session_state.video_id=video_id
            st.write(f"Video ID: {video_id}")
            try:
                ytt=get_youtube_api()
                yt2=ytt.list(video_id)
                transcript_list=ytt.fetch(video_id=video_id).to_raw_data()
                transcript=" ".join(chunk["text"] for chunk in transcript_list)
                st.session_state.transcript = transcript
                st.success("Transcript Landed")
                splitter=RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                chunks=splitter.create_documents([transcript])
                embeddings=OllamaEmbeddings(model="mxbai-embed-large")
                vector_store=FAISS.from_documents(chunks,embeddings)
                st.session_state.vector_store=vector_store
            except TranscriptsDisabled:
                print("NO CAPTION available for this video")
                st.session_state.transcript = ""
            except Exception as e:
                st.error(f"An error occurred : {e}")
        else:
            st.error("Please enter a valid youtube url")


if (st.session_state.transcript and st.session_state.vector_store):
    st.divider()
    st.subheader("Q n A  :)")
    retriever=st.session_state.vector_store.as_retriever(search_type="similarity",search_kwargs={'k':4})
    model=OllamaLLM(model="llama3")
    prompt=PromptTemplate(
        template="""You are a helpful asistant
        Answer only from the provided transcript context
        if the context is insufficient, just answer i don't know,
        keep the answer precised
        here is the context {context}
        and here is the question asked {question}
        """,
        input_variables=['context','question'])
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    user_input=st.chat_input("type any question")
    if(user_input):
        st.chat_message("user").markdown(user_input)
        st.session_state.chat_history.append({"role":"user","content":user_input})
        result=retriever.invoke(user_input)
        context_text="\n\n".join(doc.page_content for doc in result)
        final_prompt=prompt.invoke({"context":context_text,"question":user_input})
        ans=model.invoke(final_prompt)
        with st.chat_message("assistant"):
                st.markdown(ans)
        st.session_state.chat_history.append({"role": "assistant", "content": ans})
else:
    st.info("Enter the youtube link in the sidebar")
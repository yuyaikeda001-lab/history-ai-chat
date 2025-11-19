import streamlit as st
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pinecone import Pinecone

# ページ設定
st.set_page_config(page_title="歴史人物AI", page_icon="📜")

st.title("📜 歴史人物AIチャット")

# サイドバー：APIキーの設定（ローカル実行用。デプロイ時はSecretsを使います）
with st.sidebar:
    st.header("設定")
    st.markdown("PDFをアップロードすると、その内容を学習して賢くなります。")
    uploaded_file = st.file_uploader("学習させるPDF", type="pdf")

# APIキーの取得（Streamlit Secrets または 環境変数）
openai_api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
pinecone_api_key = st.secrets.get("PINECONE_API_KEY") or os.getenv("PINECONE_API_KEY")
pinecone_index_name = "history-chat" # Pineconeで作ったIndex名

if not openai_api_key or not pinecone_api_key:
    st.error("APIキーが設定されていません。StreamlitのSecretsに設定してください。")
    st.stop()

# 環境変数のセット
os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["PINECONE_API_KEY"] = pinecone_api_key

# --- 1. PDFの学習処理（賢くなる部分） ---
if uploaded_file is not None:
    with st.spinner("資料を読み込んで記憶しています..."):
        # 一時ファイルとして保存
        temp_pdf_path = "temp_uploaded.pdf"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # PDF読み込みと分割
        loader = PyPDFLoader(temp_pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)
        
        # Pineconeへ保存（永続記憶）
        embeddings = OpenAIEmbeddings()
        PineconeVectorStore.from_documents(docs, embeddings, index_name=pinecone_index_name)
        
        st.success(f"{len(docs)} ページ分の知識を獲得しました！")
        os.remove(temp_pdf_path) # 掃除

# --- 2. チャットエンジンの準備 ---
@st.cache_resource
def get_chat_chain():
    # Pineconeから知識を取り出す設定
    embeddings = OpenAIEmbeddings()
    vectorstore = PineconeVectorStore(index_name=pinecone_index_name, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    
    # AIの設定（歴史上の人物になりきるプロンプトはここで調整）
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)
    
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True,
        output_key="answer"
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=False,
        verbose=True
    )
    return chain

chain = get_chat_chain()

# --- 3. チャット画面の表示 ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "某（それがし）に何か用か？"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

prompt = st.chat_input("質問を入力...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner("思案中..."):
            response = chain.invoke({"question": prompt})
            answer = response["answer"]
            st.write(answer)
            
    st.session_state.messages.append({"role": "assistant", "content": answer})
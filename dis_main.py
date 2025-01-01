import os, char_ratio, log_util
import discord
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains import RetrievalQA

# DB 저장 디렉토리 생성
DB_DIR = "./db"
if not os.path.exists(DB_DIR):
    os.makedirs(DB_DIR)

load_dotenv(override=True)
TOKEN = os.getenv('DISCORD_TOKEN')
base_url = os.getenv('BASE_URL')
intents = discord.Intents.all()
client = discord.Client(command_prefix='!', intents=intents)
high_level_model = ChatOllama(base_url=base_url, model="gemma2:27b-instruct-q6_K", temperature=0.1, num_predict=32768)
chain = char_ratio.get_char_chain(high_level_model)

def log(msg):
    log_util.log(msg,save_to_file=True)

async def get_or_create_retriever(channel_id, message, new_message = None):
    embedding_model = "snowflake-arctic-embed2:latest"
    embeddings = OllamaEmbeddings(base_url=base_url, model=embedding_model)
    
    db_path = os.path.join(DB_DIR, f"channel_{channel_id}")
    
    if os.path.exists(db_path):
        log(f"기존 DB 로드: 채널 ID = {channel_id}")
        chroma_store = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )
        if new_message:
            doc_content = f"{new_message['sender']}: {new_message['content']}"
            metadata = {"channel_id": channel_id, "sender": new_message['sender']}
            text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
            chunks = text_splitter.split_text(doc_content)
            chroma_store.add_texts(
                texts=chunks,
                metadatas=[metadata] * len(chunks)
            )
            chroma_store.persist()
    else:
        log(f"새 DB 생성: 채널 ID = {channel_id}")
        messages_array = []
        async for msg in message.channel.history(limit=100):  # Changed to async for
            messages_array.append({
                "sender": msg.author.name,
                "content": msg.content,
                "date": msg.created_at.isoformat()
            })
        documents = []
        for msg in messages_array:
            doc_content = f"{msg['sender']}: {msg['content']}"
            doc = {"text": doc_content, "metadata": {"channel_id": channel_id, "sender": msg['sender']}}
            documents.append(doc)
        
        text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
        docs = [text_splitter.split_text(doc["text"]) for doc in documents]
        
        chroma_store = Chroma.from_texts(
            texts=[chunk for sublist in docs for chunk in sublist],
            embedding=embeddings,
            metadatas=[doc["metadata"] for doc in documents],
            persist_directory=db_path
        )
        chroma_store.persist()
    
    return chroma_store.as_retriever()

@client.event
async def on_ready():
    print(f'We have logged in as {client.user.name}')

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    
    current_message = {
        "sender": message.author.name,
        "content": message.content
    }
    retriever = await get_or_create_retriever(message.channel.id, message, current_message)
    
    if message.content.startswith("#레이시오"):
        await message.channel.send("(답변 작성중)")
        relevant_docs = retriever.get_relevant_documents(message.content)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        
        # Add context to the chain input
        chain_input = {
            "user_input": message.content,
            "past_chat": context
        }
        log(f"Chain Input: {chain_input}")
        response = chain.invoke(chain_input)
        await message.channel.send(response)

client.run(TOKEN)

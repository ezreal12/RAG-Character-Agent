import os
from datetime import datetime, timedelta
import asyncio
import aiohttp
import backoff
import discord
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from news import news_db, crol_utils, news_summury_module
import char_ratio
import log_util

# 환경 변수 로드
load_dotenv(override=True)
TOKEN = os.getenv('DISCORD_TOKEN')
BASE_URL = os.getenv('BASE_URL')
NEWS_RUN_TIME = os.getenv('NEWS_RUN_TIME',"18:00")
DB_DIR = "./db"

# 디렉토리 생성
os.makedirs(DB_DIR, exist_ok=True)

# Discord 클라이언트 초기화
intents = discord.Intents.all()
intents.guilds = True
intents.messages = True
intents.message_content = True
channel_list = []
# 모델 초기화
char_level_model = ChatOllama(
    base_url=BASE_URL, model="gemma2:27b-instruct-q6_K",
    temperature=0.1, num_predict=32768
)
high_level_model = ChatOllama(
    base_url=BASE_URL, model="gemma2:27b-instruct-q6_K",
    temperature=0, num_predict=32768
)
low_level_model = ChatOllama(base_url=BASE_URL, model="gemma2:2b", temperature=0)
chain = char_ratio.get_char_chain(char_level_model)
news_char_agent = char_ratio.get_char_news_chain(high_level_model)
error_chat_agent = char_ratio.get_char_error_chain(high_level_model)
summury_agent = news_summury_module.get_summury_chain(low_level_model)
def log(msg):
    """로그를 파일에 저장."""
    log_util.log(msg, save_to_file=True)
    
class CustomDiscordClient(discord.Client):
    @backoff.on_exception(
        backoff.expo,
        (aiohttp.ClientError, discord.errors.ConnectionClosed),
        max_tries=5
    )
    async def setup_hook(self):
        """Initialize connection with retry logic"""
        await super().setup_hook()

    async def close(self):
        """Clean shutdown of client"""
        try:
            await super().close()
        except Exception as e:
            log(f"Error during client shutdown: {str(e)}")
            
client = CustomDiscordClient(command_prefix='!', intents=intents)

async def get_or_create_retriever(channel_id, message, new_message=None):
    """DB에서 리트리버를 생성하거나 기존 리트리버를 가져옴."""
    embedding_model = "snowflake-arctic-embed2:latest"
    embeddings = OllamaEmbeddings(base_url=BASE_URL, model=embedding_model)
    db_path = os.path.join(DB_DIR, f"channel_{channel_id}")

    if os.path.exists(db_path):
        log(f"기존 DB 로드: 채널 ID = {channel_id}")
        chroma_store = Chroma(
            persist_directory=db_path, embedding_function=embeddings
        )
        if new_message:
            text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
            chunks = text_splitter.split_text(f"{new_message['sender']}: {new_message['content']}")
            metadata = {"channel_id": channel_id, "sender": new_message['sender']}
            chroma_store.add_texts(texts=chunks, metadatas=[metadata] * len(chunks))
            chroma_store.persist()
    else:
        log(f"새 DB 생성: 채널 ID = {channel_id}")
        async for msg in message.channel.history(limit=100):
            content = f"{msg.author.name}: {msg.content}"
            metadata = {"channel_id": channel_id, "sender": msg.author.name}
            text_splitter = CharacterTextSplitter(chunk_size=100, chunk_overlap=10)
            chunks = text_splitter.split_text(content)
            documents = [{"text": chunk, "metadata": metadata} for chunk in chunks]

            chroma_store = Chroma.from_texts(
                texts=[doc["text"] for doc in documents],
                embedding=embeddings,
                metadatas=[doc["metadata"] for doc in documents],
                persist_directory=db_path
            )
            chroma_store.persist()
    
    return chroma_store.as_retriever()


async def send_all_channels_message(message):
    """모든 채널에 메시지 전송."""
    global channel_list
    for channel in channel_list:
        await channel.send(message)

async def start_timer_function():
    """매일 정해진 시각에 실행되는 작업."""
    try:
        global channel_list
        if len(channel_list) == 0:
            log("채널 리스트 없음.")
            return
           
        otaku_news_data = crol_utils.get_otaku_news_data()
        ai_news_data = crol_utils.get_ai_times_data()
        crol_data = otaku_news_data + ai_news_data
        
        log(f"크롤링 데이터 수신 완료: {len(crol_data)}개의 데이터")
        for channel in channel_list:
            try:
                print(f"채널 ID: {channel.id}")
                db_path = f"./news_db/{channel.id}.json"
                
                # 디렉토리 존재 확인 및 생성
                os.makedirs(os.path.dirname(db_path), exist_ok=True)
                log(f"DB 경로 확인 및 생성 완료: {db_path}")
                
                new_data_list = news_db.insert_unique_return_new(db_path, crol_data)
                log(f"새로운 데이터 수: {len(new_data_list)}")
                for data in new_data_list:
                    try:
                        site_url = "\n"+data['site_url']
                        content = data['content']
                        log(f"뉴스 콘텐츠 처리 시작: {data}...")  # 첫 30자만 로그에 표시
                        summury_content = summury_agent.invoke({"news_article": content})
                        char_news_comment = news_char_agent.invoke({"news_article": summury_content})
                        char_news_comment = (char_news_comment + site_url)
                        await channel.send(char_news_comment)
                        log(f"뉴스 댓글 전송 완료: 채널 ID {channel.id}")
                    except Exception as e:
                        error_msg = f"뉴스 처리 중 에러 발생: {str(e)}"
                        log(error_msg)
                        await channel.send(f"뉴스 처리 중 문제가 발생했습니다: {str(e)}")
                        
            except discord.errors.Forbidden:
                log(f"채널 {channel.id}에 메시지를 보낼 권한이 없습니다.")
            except Exception as e:
                log(f"채널 {channel.id} 처리 중 에러 발생: {str(e)}")
                
    except Exception as e:
        error_msg = f"start_timer_function 실행 중 에러 발생: {str(e)}"
        log(error_msg)
        # 에러 발생 시 캐릭터 에이전트를 통한 에러 메시지 전송
        try:
            char_err_comment = error_chat_agent.invoke({"error_log": error_msg})
            await send_all_channels_message(char_err_comment)
            log("에러 메시지 전송 완료")
        except Exception as send_err:
            log(f"에러 메시지 전송 실패: {str(send_err)}")

async def run_scheduled_task():
    """매일 지정된 시간에 start_timer_function을 실행하는 비동기 스케줄러"""
    while True:
        try:
            # 현재 시간과 다음 실행 시간 계산
            now = datetime.now()
            run_time = datetime.strptime(NEWS_RUN_TIME, "%H:%M").time()
            next_run = datetime.combine(now.date(), run_time)
            
            # 이미 오늘의 실행 시간이 지났다면 다음 날로 설정
            if now.time() > run_time:
                next_run += timedelta(days=1)
            
            # 다음 실행까지 대기
            wait_seconds = (next_run - now).total_seconds()
            await asyncio.sleep(wait_seconds)
            
            # start_timer_function 실행
            await start_timer_function()
            
            # 다음 날까지 대기
            await asyncio.sleep(24 * 60 * 60)
            
        except Exception as e:
            error_msg = f"스케줄러 실행 중 오류 발생: {str(e)}"
            log(error_msg)
            await asyncio.sleep(60)  # 오류 발생 시 1분 후 재시도


@client.event
async def on_ready():
    """봇이 준비되었을 때 호출."""
    client.loop.create_task(run_scheduled_task())
    global channel_list
    # await channel.send("안녕하세요! 봇 테스트 메시지입니다.")
    # 봇이 참가한 모든 서버와 텍스트 채널 정보 조회
    for guild in client.guilds:
        print(f"서버 이름: {guild.name}, 서버 ID: {guild.id}")
        channel_list = guild.text_channels

@client.event
async def on_message(message):
    """메시지 이벤트 처리."""
    if message.author == client.user:
        return

    retriever = await get_or_create_retriever(message.channel.id, message, {
        "sender": message.author.name,
        "content": message.content
    })

    if message.content.startswith("#레이시오"):
        await message.channel.send("(답변 작성중)")
        relevant_docs = retriever.get_relevant_documents(message.content)
        context = "\n".join([doc.page_content for doc in relevant_docs])
        response = chain.invoke({"user_input": message.content, "past_chat": context})
        await message.channel.send(response)
        
@client.event
async def on_error(event, *args, **kwargs):
    error_log = f"{event}"
    log(error_log)
    char_err_comment = error_chat_agent.invoke({"error_log": error_log})
    await send_all_channels_message(char_err_comment)

if __name__ == "__main__":
    # Discord 봇 실행
    client.run(TOKEN)

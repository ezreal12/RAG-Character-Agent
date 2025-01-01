# bot.py
import os, char_ratio
import discord
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama

# 환경 변수를 .env 파일에서 로딩
load_dotenv(override=True)
TOKEN = os.getenv('DISCORD_TOKEN')
base_url = os.getenv('BASE_URL')
intents = discord.Intents.all()
client = discord.Client(command_prefix='!', intents=intents)
high_level_model = ChatOllama(base_url=base_url, model="gemma2:27b-instruct-q6_K", temperature=0, num_predict = 32768)
chain = char_ratio.get_char_chain(high_level_model)
# TODO!! 캐릭터 성격을 나타내는 아이템 (시스템 프롬프트, RAG 파일 경로 등)은 env로 뺄것
# env가 아니고 텍스트 파일로 빼야겠는데..? 저만한길이가 env에 들어감?
# git에 캐릭터마다 프로젝트 1개씩 만들지말고 서버 내부에서 디렉토리만 여러개 복사하고 env만 바꾸고, 여러개 돌리라는뜻이다.
@client.event
async def on_ready():
  print(f'We have logged in as {client.user.name}')
  
  
@client.event
async def on_message(message):
    if message.author == client.user:
        return  # bot 스스로가 보낸 메세지는 무시
    if message.content.startswith("#레이시오"):
        await message.channel.send("(답변 작성중)")
        print(f"message.content : {message.content}")
        response = chain.invoke({"user_input": message.content})
        await message.channel.send(response)

# start the bot
client.run(TOKEN)
import os
import discord
import asyncio
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()
TOKEN = os.getenv('DISCORD_TOKEN')

# Intents 설정
intents = discord.Intents.default()
intents.guilds = True
intents.messages = True
intents.message_content = True

# Discord 클라이언트 초기화
client = discord.Client(intents=intents)

# 메시지 전송을 위한 이벤트 루프와 큐
message_queue = asyncio.Queue()

async def send_notification():
    while True:
        if not message_queue.empty():
            await message_queue.get()
            for guild in client.guilds:
                for channel in guild.text_channels:
                    try:
                        await channel.send("알림: 사용자가 입력을 전송했습니다!")
                    except Exception as e:
                        print(f"채널 {channel.name}에 메시지를 보낼 수 없습니다: {e}")
        await asyncio.sleep(0.1)

async def get_input():
    while True:
        user_input = await client.loop.run_in_executor(None, input, "메시지를 보내려면 A를 입력하세요: ")
        if user_input.lower() == 'a':
            await message_queue.put(True)

@client.event
async def on_ready():
    print(f'Logged in as {client.user.name}')
    # 입력 모니터링과 알림 전송 태스크 시작
    client.loop.create_task(send_notification())
    client.loop.create_task(get_input())

@client.event
async def on_error(event, *args, **kwargs):
    print(f"에러 발생: {event}")

# 봇 실행
client.run(TOKEN)

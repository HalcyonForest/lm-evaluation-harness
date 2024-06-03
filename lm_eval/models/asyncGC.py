import asyncio
import time
import gigachat
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole
import random
import os
import re
from gigachat_model import GigaChatLM
# from gigachat_model import gigachat_completion
from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

api_key = "ZTZkM2ZmODYtNDRmNC00OWQ0LTkyNTUtOTA1NzE1ZGY2ZTFjOjJmMmU0NDZiLTk1MzItNGRlNC05ZGY0LTU2MzI4NzJkZDM4Zg=="
# os.environ[''] = creds
os.environ['GIGACHAT_API_KEY'] = api_key
print(f'\n {os.environ.get("GIGACHAT_API_KEY")}\n')

gc = GigaChatLM()

messages=[]
system = "Ты самый лучший в мире писатель детских сказок "
prompt = "Напиши длинную историю про лунтика размеров в 15 предложений"
if system:
    messages.append(
        gigachat.models.Messages(
            role=gigachat.models.MessagesRole.SYSTEM,
            content=system,
        )  
    )
    
messages.append(
        gigachat.models.Messages(
            role=gigachat.models.MessagesRole.USER,
            content=prompt,
        )
        )
print(f'Messages: {messages}')

payload = gigachat.models.Chat(
            messages=messages,
            model=gc.model,
            max_tokens_to_sample=256,
            # max_tokens=max_tokens,
            # temperature=0.8,
            # update_display=0.01,
            update_interval=0.1,
        )

stopwords = ['Лунтик'] # '<s1>','<a2>', '<r6>'
# TODO: засунуть это в функцию в модели чтобы брбрбр делало и работало. 
async def main():
    total_text = ''
    pattern = '|'.join(stopwords)
    start = time.time()
    async with GigaChat(credentials=os.environ.get("GIGACHAT_API_KEY"), scope="GIGACHAT_API_CORP", verify_ssl_certs=False) as giga:
        # stop = ['']
        async for chunk in giga.astream(payload):
            p = random.random()
            text = chunk.choices[0].delta.content
            # if p > 0.8:
            #     text += random.choice(stopwords)
            total_text += text
            print(f'Total text: {total_text}')
            k=re.search(pattern, total_text)
            # k=re.findall(pattern, total_text)
            if k:
                print(time.time(), k.span())
                break

            print(f'{time.time()} {text}', flush=True)
            # print('-')
            # print(f'{time.time()} {dir(chunk.choices[0].delta)}', flush=True)
            # break
    if k:
        goodText = total_text[:k.span()[1]]
        print(goodText)
    print(f'Total time: {round(time.time()-start,4)}')

# asyncio.run(main())

import re

async def chunk_gigachat_completion(
    client, #: gigachat.GigaChat,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    system: str,
    stopwords: list[str],
    **kwargs,
) -> str:
    try:
        import gigachat
    except ModuleNotFoundError:
        raise Exception(
            "attempted to use 'gigachat' LM type, but package `gigachat` is not installed. \
please install gigachat via `pip install 'gigachat'`",
        )
    messages=[]
    if system:
        messages.append(
            gigachat.models.Messages(
                role=gigachat.models.MessagesRole.SYSTEM,
                content=system,
            )  
        )
        
    messages.append(
            gigachat.models.Messages(
                role=gigachat.models.MessagesRole.USER,
                content=prompt,
            )
            )
    print(f'Messages: {messages}')

    async def completion():
        
        payload = gigachat.models.Chat(
            messages=messages,
            model=model,
            max_tokens_to_sample=max_tokens,
            # max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
            update_display=0.1
        )

        # response = client.chat(
        #     payload
        # )

        total_text = ''
        pattern = '|'.join(stopwords)
        # start = time.time()
        async with client as giga:
        # stop = ['']
            async for chunk in giga.astream(payload):
                text = chunk.choices[0].delta.content
                total_text += text
                print(f'Total text: {total_text}')
                k=re.search(pattern, total_text)
                if k:
                    break


        if k:
            goodText = total_text[:k.span()[0]]
            print(goodText)

        return total_text
    
    answer = await completion()
    return answer

if __name__ == "__main__":
    asyncio.run(chunk_gigachat_completion(gc.client,gc.model,'привет. Как твои дела? Какая сейчас погода в Люксембурге. QQQ',256,0.8,'переведи на английский. В конце добавь "QQQ"',['QQQ']))
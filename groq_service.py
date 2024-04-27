from groq import Groq
import os

from dotenv import load_dotenv, find_dotenv

# read local .env file
_ = load_dotenv(find_dotenv()) 

client = Groq(
    api_key=os.environ['GROQ_API_KEY']
    )

def groq(prompt):
    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {
                "role": "system",
                "content": "你是一个龙与地下城的玩家， 你用中文回答所有问题"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )
    response = ""
    for chunk in completion:
        response += chunk.choices[0].delta.content or ""
    print(response)
    return response

if __name__ == "__main__":
    print(groq("人生的意义是什么"))
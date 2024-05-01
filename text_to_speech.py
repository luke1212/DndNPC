import os
import torch
from pathlib import Path
import openai
from dotenv import load_dotenv
from bark import SAMPLE_RATE, generate_audio, preload_models
from scipy.io.wavfile import write as write_wav
from IPython.display import Audio
from TTS.api import TTS

from deepgram import (
    DeepgramClient,
    SpeakOptions,
)

load_dotenv()
filename = "pizza.wav"
os.environ["SUNO_USE_SMALL_MODELS"] = "True"
openai.api_key = os.environ['OPENAI_API_KEY']


def openai_text_to_speech(text):

    client = openai.OpenAI()
    
    with client.audio.speech.with_streaming_response.create(
    model="tts-1-hd",
    voice="alloy",
    input=text,
    ) as response:
        response.stream_to_file("speech.mp3")
    return "speech.mp3"


if __name__ == "__main__":
    openai_text_to_speech("你好我是孙庚. 我是赵蓓的义父")
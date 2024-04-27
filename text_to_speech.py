import os
import torch
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

def tts_text_to_speech(text):
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load the model to GPU
    # Bark is really slow on CPU, so we recommend using GPU.
    tts = TTS("tts_models/multilingual/multi-dataset/bark").to(device)
    # Cloning a new speaker
    # This expects to find a mp3 or wav file like `bark_voices/new_speaker/speaker.wav`
    # It computes the cloning values and stores in `bark_voices/new_speaker/speaker.npz`
    tts.tts_to_file(text=text,
                    file_path="tts.wav",
                    speaker="ljspeech")

def bark_text_to_speech(text):
    
    # download and load all models
    preload_models()

    audio_array = generate_audio(text, history_prompt = "v2/zh_speaker_1")

    # save audio to disk
    write_wav("bark_generation.wav", SAMPLE_RATE, audio_array)
    
    # play text in notebook
    Audio(audio_array, rate=SAMPLE_RATE)

def text_to_speech(text):
    try:
        SPEAK_OPTIONS = {"text": text}
        # STEP 1: Create a Deepgram client using the API key from environment variables
        deepgram = DeepgramClient(api_key=os.getenv("DG_API_KEY"))

        # STEP 2: Configure the options (such as model choice, audio configuration, etc.)
        options = SpeakOptions(
            model="aura-asteria-en",
            encoding="linear16",
            container="wav"
        )

        # STEP 3: Call the save method on the speak property
        response = deepgram.speak.v("1").save(filename, SPEAK_OPTIONS, options)
        return filename

    except Exception as e:
        print(f"Exception: {e}")


if __name__ == "__main__":
    tts_text_to_speech("你好我是孙庚. 我是赵蓓的义父")
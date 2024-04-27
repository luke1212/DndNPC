import os
from dotenv import load_dotenv

from deepgram import (
    DeepgramClient,
    PrerecordedOptions,
    FileSource,
)

load_dotenv()
API_KEY = os.getenv("DG_API_KEY")


def speech_to_text(audio_file):
    try:
        # STEP 1 Create a Deepgram client using the API key
        deepgram = DeepgramClient(API_KEY)

        with open(audio_file, "rb") as file:
            buffer_data = file.read()

        payload: FileSource = {
            "buffer": buffer_data,
        }

        #STEP 2: Configure Deepgram options for audio analysis
        options = PrerecordedOptions(
            model="base",
            smart_format=True,
            language="zh",
        )

        # STEP 3: Call the transcribe_file method with the text payload and options
        response = deepgram.listen.prerecorded.v("1").transcribe_file(payload, options)

        transcript = response["results"]['channels'][0]['alternatives'][0]['transcript']
        return transcript

    except Exception as e:
        print(f"Exception: {e}")


if __name__ == "__main__":
    print(speech_to_text("pizza.wav"))

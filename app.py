import gradio as gr
import speech_to_text as speech_to_text
import text_to_speech as text_to_speech
import groq_service as groq_service

demo = gr.Blocks()
groq_chain = groq_service.lang_chain_groq()

def chat_groq(audio_file):
    transcript = speech_to_text.speech_to_text(audio_file)
    groq_response = groq_chain.invoke(transcript)
    print("History:\n\n" + groq_response["chat_history"])
    print("Response:\n\n" + groq_response["text"])
    return text_to_speech.openai_text_to_speech(groq_response["text"])

if __name__ == "__main__":
   
    mic_transcribe = gr.Interface(
    fn=chat_groq,
    inputs=gr.Audio(sources="microphone",
                    type="filepath"),
    outputs=gr.Audio(label="Transcription",
                       type="filepath"),
    allow_flagging="never")

    file_transcribe = gr.Interface(
        fn=chat_groq,
        inputs=gr.Audio(sources="upload",
                        type="filepath"),
        outputs=gr.Audio(label="Transcription",
                        type="filepath"),
        allow_flagging="never",
    )
    
    gpt = gr.Interface(
        fn=speech_to_text.speech_to_text,
        inputs=gr.Audio(sources="upload",
                        type="filepath"),
        outputs=gr.Textbox(label="Transcription",
                        lines=3),
        allow_flagging="never",
    )
    
    with demo:
        gr.TabbedInterface(
            [mic_transcribe,
            file_transcribe,
            gpt],
            ["Transcribe Microphone",
            "Transcribe Audio File",
            "GPT"],
        )
        demo.launch(share=True, 
            server_port= 8001)
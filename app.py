import gradio as gr
import speech_to_text as speech_to_text
import text_to_speech as text_to_speech
import ai_service as ai_service

demo = gr.Blocks()
_groq_chain = ai_service.lang_chain_groq()
_openai_agent = ai_service.openai_agent()

def chat_groq(audio_file):
    transcript = speech_to_text.speech_to_text(audio_file)
    response = _openai_agent.invoke({"input": transcript})
    print(response['output'])
    print('History:', '\n'.join([str(lst) for lst in response['chat_history']]))
    return text_to_speech.openai_text_to_speech(response['output'])

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
        
# if __name__ == "__main__":
#    print(_openai_agent.invoke({"input": "你好我叫luke 很高兴认识你"})['output'])
#    print('History:', '\n'.join([str(lst) for lst in _openai_agent.invoke({"input": "你好我叫luke 很高兴认识你"})['chat_history']]))
   
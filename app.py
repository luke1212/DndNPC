import gradio as gr
import speech_to_text as speech_to_text
import text_to_speech as text_to_speech
import ai_service as ai_service

demo = gr.Blocks()
_groq_chain = ai_service.lang_chain_groq()
_openai_agent = ai_service.openai_agent()
_chat_with_url = ai_service.chat_with_url("https://react.dev/learn", 2)

def chat_groq(audio_file):
    transcript = speech_to_text.speech_to_text(audio_file)n
    response = _openai_agent.invoke({"input": transcript})
    print(response['output'])
    print('History:', '\n'.join([str(lst) for lst in response['chat_history']]))
    return text_to_speech.openai_text_to_speech(response['output'])

def chat_with_url(question):
    response = _chat_with_url.invoke(question)
    metadata = ""
    for i in range(1, len(response['source_documents'])):
        metadata += '\n' + response['source_documents'][i].metadata['source'] + ' \n'  + response['source_documents'][i].metadata['title']
    response = response['result'] + '\n' + metadata
    return response

if __name__ == "__main__":
   
    mic_transcribe = gr.Interface(
    fn=chat_groq,
    inputs=gr.Audio(sources="microphone",
                    type="filepath"),
    outputs=gr.Audio(label="Transcription",
                       type="filepath"),
    allow_flagging="never")

    file_transcribe = gr.Interface(
        fn=chat_with_url,
        inputs=gr.Textbox(label="Transcription",
                        lines=3),
        outputs=gr.Textbox(label="Transcription",
                        lines=3),
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
#     print(chat_with_url("Dictionary Merge & Update Operators"))
   
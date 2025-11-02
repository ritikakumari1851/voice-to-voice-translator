import os
import numpy as np
import gradio as gr
import assemblyai as aai
from translate import Translator
import uuid
from typing import List
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from pathlib import Path


def voice_to_voice(audio_file):
    # Transcribe
    transcript = transcribe_audio(audio_file)
    if transcript.status == aai.TranscriptStatus.error:
        raise gr.Error(str(transcript.error or "Unknown error"))
    else:
        transcript = transcript.text

    # Translate
    list_translations: list[str] = translate_text(transcript)
    generated_audio_paths = []

    # Convert text → speech
    for translation in list_translations:
        translated_audio_file_name = text_to_speech(translation)
        path = Path(translated_audio_file_name)
        generated_audio_paths.append(path)

    return (
        generated_audio_paths[0], generated_audio_paths[1], generated_audio_paths[2],
        generated_audio_paths[3], generated_audio_paths[4], generated_audio_paths[5],
        list_translations[0], list_translations[1], list_translations[2],
        list_translations[3], list_translations[4], list_translations[5]
    )


def transcribe_audio(audio_file):
    aai.settings.api_key = "41f68bdcaa9542c2a7f2bdf3f71b60fc"
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_file)
    return transcript


def translate_text(text: str) -> List[str]:
    # Updated languages
    languages = ["ru", "de", "es", "ja", "hi", "ta"]
    list_translations = []

    for lan in languages:
        translator = Translator(from_lang="en", to_lang=lan)
        translation = translator.translate(text)
        list_translations.append(translation)

    return list_translations


def text_to_speech(text: str) -> str:
    client = ElevenLabs(
        api_key="sk_0f54d35e52601e79e917547fb15d5ab87054af6cc33eb304",
    )

    response = client.text_to_speech.convert(
        voice_id="MzCUHSlV6TI9TcOa1GTx",
        model_id="eleven_multilingual_v2",
        text=text,
        voice_settings=VoiceSettings(
            stability=0.5,
            similarity_boost=0.8,
            style=0.5,
            use_speaker_boost=True,
        ),
        output_format="mp3_22050_32",
    )

    save_file_path = f"{uuid.uuid4()}.mp3"
    with open(save_file_path, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)

    print(f"{save_file_path}: Audio saved successfully!")
    return save_file_path


with gr.Blocks() as demo:
    gr.Markdown("## 🎤 Record yourself in English and receive translations instantly!")

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                show_download_button=True,
                waveform_options=gr.WaveformOptions(
                    waveform_color="#01C6FF",
                    waveform_progress_color="#0066B4",
                    skip_length=2,
                    show_controls=False,
                ),
            )
            with gr.Row():
                submit = gr.Button("Submit", variant="primary")
                gr.Button("Clear").click(lambda: None, None, audio_input)

    # --- First row ---
    with gr.Row():
        with gr.Group():
            ru_output = gr.Audio(label="Russian", interactive=False)
            ru_text = gr.Markdown()
        with gr.Group():
            de_output = gr.Audio(label="German", interactive=False)
            de_text = gr.Markdown()
        with gr.Group():
            es_output = gr.Audio(label="Spanish", interactive=False)
            es_text = gr.Markdown()

    # --- Second row ---
    with gr.Row():
        with gr.Group():
            jp_output = gr.Audio(label="Japanese", interactive=False)
            jp_text = gr.Markdown()
        with gr.Group():
            hi_output = gr.Audio(label="Hindi", interactive=False)
            hi_text = gr.Markdown()
        with gr.Group():
            ta_output = gr.Audio(label="Tamil", interactive=False)
            ta_text = gr.Markdown()

    output_components = [
        ru_output, de_output, es_output, jp_output, hi_output, ta_output,
        ru_text, de_text, es_text, jp_text, hi_text, ta_text
    ]

    submit.click(fn=voice_to_voice, inputs=audio_input, outputs=output_components)

if __name__ == "__main__":
    import os
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))

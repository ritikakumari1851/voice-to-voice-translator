import gradio as gr
import assemblyai as aai
from translate import Translator
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
import uuid
from pathlib import Path


def voice_to_voice(audio_file):
    # Step 1: Transcribe audio
    transcription_response = audio_transcription(audio_file)

    if transcription_response.status == aai.TranscriptStatus.error:
        raise gr.Error(str(transcription_response.error or "Unknown error"))
    else:
        text = transcription_response.text

    # Step 2: Translate text into 4 languages
    es_translation, hi_translation, ja_translation, ta_translation = text_translation(text)

    # Step 3: Convert translated text into audio
    es_audio_path = text_to_speech(es_translation)
    hi_audio_path = text_to_speech(hi_translation)
    ja_audio_path = text_to_speech(ja_translation)
    ta_audio_path = text_to_speech(ta_translation)

    # Step 4: Return paths for Gradio output
    return Path(es_audio_path), Path(hi_audio_path), Path(ja_audio_path), Path(ta_audio_path)


def audio_transcription(audio_file):
    aai.settings.api_key = "41f68bdcaa9542c2a7f2bdf3f71b60fc"
    transcriber = aai.Transcriber()
    transcription = transcriber.transcribe(audio_file)
    return transcription


def text_translation(text):
    translator_es = Translator(from_lang="en", to_lang="es")
    es_text = translator_es.translate(text)

    translator_hi = Translator(from_lang="en", to_lang="hi")
    hi_text = translator_hi.translate(text)

    translator_ja = Translator(from_lang="en", to_lang="ja")
    ja_text = translator_ja.translate(text)

    translator_ta = Translator(from_lang="en", to_lang="ta")
    ta_text = translator_ta.translate(text)

    return es_text, hi_text, ja_text, ta_text


def text_to_speech(text):
    client = ElevenLabs(
        api_key="sk_0f54d35e52601e79e917547fb15d5ab87054af6cc33eb304",
    )

    response = client.text_to_speech.convert(
        voice_id="MzCUHSlV6TI9TcOa1GTx",
        optimize_streaming_latency="0",
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_multilingual_v2",
        voice_settings=VoiceSettings(
            stability=0.5,
            similarity_boost=0.8,
            style=0.5,
            use_speaker_boost=True,
        ),
    )

    save_file_path = f"{uuid.uuid4()}.mp3"
    with open(save_file_path, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)

    print(f"{save_file_path}: A new audio file was saved successfully!")
    return save_file_path


audio_input = gr.Audio(
    sources=["microphone"],
    type="filepath"
)

demo = gr.Interface(
    fn=voice_to_voice,
    inputs=audio_input,
    outputs=[
        gr.Audio(label="Spanish"),
        gr.Audio(label="Hindi"),
        gr.Audio(label="Japanese"),
        gr.Audio(label="Tamil")
    ]
)

if __name__ == "__main__":
    demo.launch()

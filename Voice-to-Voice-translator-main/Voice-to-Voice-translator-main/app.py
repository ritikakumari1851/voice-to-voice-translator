import os
import gradio as gr
import assemblyai as aai
from deep_translator import GoogleTranslator
from gtts import gTTS
import uuid

# ==============================
# CONFIGURATION
# ==============================
aai.settings.api_key = "41f68bdcaa9542c2a7f2bdf3f71b60fc"

# Ensure output folder exists
os.makedirs("audio_outputs", exist_ok=True)


# ==============================
# SPEECH TO TEXT
# ==============================
def speech_to_text(audio_file: str):
    """Transcribes audio and detects language automatically using AssemblyAI."""
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_file)

    if transcript.status == aai.TranscriptStatus.error:
        return None, None, f"Transcription Error: {transcript.error}"

    text = transcript.text or "No speech detected."
    detected_lang = getattr(transcript, "language_code", "auto")

    return text, detected_lang, None


# ==============================
# TRANSLATION
# ==============================
def translate_text(text: str, target_lang: str):
    """Translates text using Deep Translator (Google)."""
    try:
        translated = GoogleTranslator(source="auto", target=target_lang).translate(text)
        return translated, None
    except Exception as e:
        return None, f"Translation Error: {e}"


# ==============================
# TEXT TO SPEECH
# ==============================
def text_to_speech(text: str, lang: str = "en"):
    """Converts text to speech using gTTS."""
    try:
        tts = gTTS(text=text, lang=lang)
        output_path = f"audio_outputs/{uuid.uuid4()}.mp3"
        tts.save(output_path)
        return output_path, None
    except Exception as e:
        return None, f"TTS Error: {e}"


# ==============================
# MAIN PIPELINE
# ==============================
def voice_to_voice(audio_input: str | None, target_lang_name: str):
    if not audio_input:
        return "Please record or upload an audio file.", None

    # Convert target language name to code
    target_lang = language_map[target_lang_name.lower()]

    print("🎙️ Transcribing audio...")
    transcript_text, detected_lang, error = speech_to_text(audio_input)
    if error:
        return error, None

    print(f"Detected language: {detected_lang}")
    print("🌐 Translating text...")
    translated_text, error = translate_text(transcript_text, target_lang)
    if error:
        return error, None

    print("🔊 Converting text to speech...")
    translated_audio, error = text_to_speech(translated_text, target_lang)
    if error:
        return error, None

    result_text = (
        f"Detected Language: {detected_lang}\n\n"
        f"Original Text: {transcript_text}\n\n"
        f"Translated Text ({target_lang_name}): {translated_text}"
    )

    return result_text, translated_audio


# ==============================
# LANGUAGE MAPPING
# ==============================
language_map = {
    "english": "en",
    "hindi": "hi",
    "tamil": "ta",
    "malayalam": "ml",
    "telugu": "te",
    "kannada": "kn",
    "bengali": "bn",
    "marathi": "mr",
    "gujarati": "gu",
    "punjabi": "pa",
    "urdu": "ur",
    "oriya": "or",
    "sanskrit": "sa",
}

# ==============================
# GRADIO INTERFACE
# ==============================
iface = gr.Interface(
    fn=voice_to_voice,
    inputs=[
        gr.Audio(sources=["microphone", "upload"], type="filepath", label="🎤 Input Audio"),
        gr.Dropdown(
            choices=list(language_map.keys()),
            label="🌍 Translate To",
            value="english",
        ),
    ],
    outputs=[
        gr.Textbox(label="📝 Translation Details"),
        gr.Audio(label="🔊 Output Speech"),
    ],
    title="🎧 AI Voice-to-Voice Translator (AssemblyAI + Deep Translator)",
    description=(
        "🎙️ Speak in any language — this AI detects your language automatically using AssemblyAI, "
        "translates it into your chosen target language, and plays the translated voice using gTTS.\n\n"
        "Supported Indian languages: Hindi, Tamil, Malayalam, Telugu, Kannada, Bengali, Marathi, Gujarati, Punjabi, Urdu, and more."
    ),
    theme="gradio/soft",
)

if __name__ == "__main__":
    iface.launch(server_name="127.0.0.1", server_port=7860)

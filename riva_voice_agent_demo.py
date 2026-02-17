"""
NVIDIA Riva Voice Agent Demo
=============================

A voice-powered AI agent using:
  - NVIDIA Riva ASR (speech-to-text) via cloud API
  - NVIDIA Riva TTS (text-to-speech) via cloud API
  - Grok 3 Fast as the conversational LLM
  - Gradio web UI for recording/uploading audio

Architecture:
  Microphone/Audio File
        |
        v
  Riva ASR (Parakeet) --> transcribed text
        |
        v
  Grok 3 Fast (LLM) --> response text
        |
        v
  Riva TTS (Magpie) --> spoken audio response

Requirements:
    pip install nvidia-riva-client gradio openai
    export NVIDIA_API_KEY="your-key"   # Free at build.nvidia.com
    export GROK_API_KEY="your-key"     # From x.ai

Usage:
    python3 riva_voice_agent_demo.py
"""

import os
import io
import json
import wave
import struct
import tempfile

import gradio as gr
import riva.client
from openai import OpenAI

# --- API Keys ---

NVIDIA_API_KEY = os.environ.get("NVIDIA_API_KEY")
if not NVIDIA_API_KEY:
    raise ValueError("Set NVIDIA_API_KEY env var — get one free at build.nvidia.com")

GROK_API_KEY = os.environ.get("GROK_API_KEY")
if not GROK_API_KEY:
    settings_path = os.path.expanduser("~/.grok/user-settings.json")
    if os.path.exists(settings_path):
        with open(settings_path) as f:
            GROK_API_KEY = json.load(f).get("apiKey")
if not GROK_API_KEY:
    raise ValueError("Set GROK_API_KEY env var or add apiKey to ~/.grok/user-settings.json")


# --- NVIDIA Riva: ASR (Speech-to-Text) ---

RIVA_ASR_FUNCTION_ID = "1598d209-5e27-4d3c-8079-4751568b1081"

asr_auth = riva.client.Auth(
    uri="grpc.nvcf.nvidia.com:443",
    use_ssl=True,
    metadata_args=[
        ["function-id", RIVA_ASR_FUNCTION_ID],
        ["authorization", f"Bearer {NVIDIA_API_KEY}"],
    ],
)
asr_service = riva.client.ASRService(asr_auth)


# --- NVIDIA Riva: TTS (Text-to-Speech) ---

RIVA_TTS_FUNCTION_ID = "877104f7-e885-42b9-8de8-f6e4c6303969"

tts_auth = riva.client.Auth(
    uri="grpc.nvcf.nvidia.com:443",
    use_ssl=True,
    metadata_args=[
        ["function-id", RIVA_TTS_FUNCTION_ID],
        ["authorization", f"Bearer {NVIDIA_API_KEY}"],
    ],
)
tts_service = riva.client.SpeechSynthesisService(tts_auth)


# --- Grok 3 Fast (Conversational LLM) ---

grok_client = OpenAI(
    api_key=GROK_API_KEY,
    base_url="https://api.x.ai/v1",
)

SYSTEM_PROMPT = """You are a helpful, concise voice assistant. Keep responses short and conversational —
the user is listening, not reading. Aim for 2-3 sentences unless the question requires more detail.
Be direct and friendly."""

conversation_history = []


# --- Core Functions ---

def transcribe_audio(audio_path):
    """Send audio to Riva ASR and get transcription."""
    if audio_path is None:
        return ""

    # Read the audio file
    with open(audio_path, "rb") as f:
        audio_data = f.read()

    # Configure ASR
    config = riva.client.RecognitionConfig(
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        max_alternatives=1,
        enable_automatic_punctuation=True,
        verbatim_transcripts=False,
        language_code="en-US",
    )
    riva.client.add_audio_file_specs_to_config(config, audio_path)

    # Call Riva ASR
    response = asr_service.offline_recognize(audio_data, config)

    transcript = ""
    for result in response.results:
        if result.alternatives:
            transcript += result.alternatives[0].transcript + " "

    return transcript.strip()


def generate_response(user_text):
    """Send text to Grok and get a response."""
    if not user_text:
        return "I didn't catch that. Could you try again?"

    conversation_history.append({"role": "user", "content": user_text})

    response = grok_client.chat.completions.create(
        model="grok-3-fast",
        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history[-10:],
        max_tokens=300,
        temperature=0.7,
    )

    assistant_text = response.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": assistant_text})

    return assistant_text


def synthesize_speech(text):
    """Send text to Riva TTS and get audio back."""
    if not text:
        return None

    # Call Riva TTS (empty voice_name uses server default)
    sample_rate = 22050
    resp = tts_service.synthesize(
        text,
        voice_name="",
        language_code="en-US",
        encoding=riva.client.AudioEncoding.LINEAR_PCM,
        sample_rate_hz=sample_rate,
    )

    # Save to WAV file
    output_path = tempfile.mktemp(suffix=".wav", dir="/tmp/claude")
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(resp.audio)

    return output_path


def process_voice(audio_path):
    """Full pipeline: ASR -> LLM -> TTS."""
    if audio_path is None:
        return "", "No audio received.", None

    # Step 1: Transcribe
    transcript = transcribe_audio(audio_path)
    if not transcript:
        return "", "Could not transcribe audio. Please try again.", None

    # Step 2: Generate response
    response_text = generate_response(transcript)

    # Step 3: Synthesize speech
    audio_output = synthesize_speech(response_text)

    return transcript, response_text, audio_output


def process_text(user_text):
    """Text input pipeline: LLM -> TTS."""
    if not user_text or not user_text.strip():
        return "", None

    # Step 1: Generate response
    response_text = generate_response(user_text.strip())

    # Step 2: Synthesize speech
    audio_output = synthesize_speech(response_text)

    return response_text, audio_output


def clear_conversation():
    """Reset conversation history."""
    conversation_history.clear()
    return "", "", None, ""


# --- Gradio Web UI ---

with gr.Blocks(title="NVIDIA Riva Voice Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # NVIDIA Riva Voice Agent
        **ASR**: Riva Parakeet (speech-to-text) | **LLM**: Grok 3 Fast | **TTS**: Riva Magpie (text-to-speech)

        Speak or type — the agent transcribes your voice, thinks with Grok, and speaks back via Riva.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input")
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",
                label="Record or upload audio",
            )
            voice_btn = gr.Button("Send Voice", variant="primary")

            gr.Markdown("---")
            text_input = gr.Textbox(
                label="Or type your message",
                placeholder="Type here and press Send Text...",
                lines=2,
            )
            text_btn = gr.Button("Send Text", variant="secondary")
            clear_btn = gr.Button("Clear Conversation")

        with gr.Column(scale=1):
            gr.Markdown("### Output")
            transcript_output = gr.Textbox(label="You said (Riva ASR)", interactive=False)
            response_output = gr.Textbox(label="Agent response (Grok 3 Fast)", interactive=False)
            audio_output = gr.Audio(label="Agent voice (Riva TTS)", type="filepath", autoplay=True)

    # Wire up events
    voice_btn.click(
        fn=process_voice,
        inputs=[audio_input],
        outputs=[transcript_output, response_output, audio_output],
    )

    text_btn.click(
        fn=process_text,
        inputs=[text_input],
        outputs=[response_output, audio_output],
    )

    clear_btn.click(
        fn=clear_conversation,
        outputs=[transcript_output, response_output, audio_output, text_input],
    )

    gr.Markdown(
        """
        ---
        **How it works**: Your voice → Riva ASR (on NVIDIA GPU) → transcribed text → Grok 3 Fast (reasoning) →
        response text → Riva TTS (on NVIDIA GPU) → spoken response. Three components, full pipeline.
        """
    )


if __name__ == "__main__":
    print("=" * 60)
    print("  NVIDIA Riva Voice Agent")
    print("  ASR: Riva Parakeet  |  LLM: Grok 3 Fast  |  TTS: Riva Magpie")
    print("=" * 60)
    demo.launch()

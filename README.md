# NVIDIA Riva Voice Agent

A voice-powered AI agent using NVIDIA Riva for speech (ASR + TTS) and Grok 3 Fast for conversational intelligence. Includes a Gradio web UI for recording/uploading audio.

## Files

- **`nvidia-riva-voice-agent-blog.md`** — Blog post covering the evolution from Jarvis to Riva and the shift from dialog management to AI agents
- **`riva_voice_agent_demo.py`** — Working demo with Gradio web UI

## Architecture

```
Microphone / Audio File
        |
        v
+---------------------------+
|   NVIDIA Riva ASR         |
|   Parakeet CTC 1.1B      |  <-- Speech-to-text (NVIDIA cloud or on-premise)
+-------------+-------------+
              |
              v
+---------------------------+
|   Grok 3 Fast (LLM)      |  <-- Conversational reasoning
|   via OpenAI-compatible   |
+-------------+-------------+
              |
              v
+---------------------------+
|   NVIDIA Riva TTS         |
|   FastPitch + HiFi-GAN    |  <-- Text-to-speech (NVIDIA cloud or on-premise)
+-------------+-------------+
              |
              v
     Spoken audio response
```

## Quick Start

```bash
pip install nvidia-riva-client gradio openai
export NVIDIA_API_KEY="your-key"   # Free at build.nvidia.com
export GROK_API_KEY="your-key"     # From x.ai
python3 riva_voice_agent_demo.py
```

Opens a web UI at `http://127.0.0.1:7860` where you can record or upload audio, see the transcription, and hear the agent's spoken response.

Get a free NVIDIA API key at [build.nvidia.com](https://build.nvidia.com).

## Key Points

- **Riva ASR** uses the Parakeet CTC 1.1B model via NVIDIA's cloud gRPC endpoint
- **Riva TTS** uses FastPitch + HiFi-GAN via NVIDIA's cloud gRPC endpoint
- **Grok 3 Fast** handles conversational reasoning between ASR and TTS
- Both Riva services can run on-premise on NVIDIA hardware for full data sovereignty
- The entire pipeline works synchronously — speak, wait, hear the response

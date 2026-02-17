# NVIDIA Riva — From Dialog Management To AI Agents, And Why Voice Is The Interface That Matters Now

## The Speech Layer That Survived The Great Rewrite

I wrote about NVIDIA Jarvis back when it was still called Jarvis. At the time, the big question was dialog management — how do you handle multi-turn conversations, track state, manage context? NVIDIA did not build their own. They integrated with **Rasa** and **Google Dialogflow** for that component.

That felt like a gap at the time. A conversational AI platform without native dialog management seemed incomplete.

Looking at it now, that gap turned out to be a feature.

---

### From Jarvis To Riva

NVIDIA announced Jarvis at **GTC 2020**. Jensen Huang demonstrated it with "Misty," a conversational weather chatbot — an end-to-end pipeline for speech recognition, language understanding, and text-to-speech. It was impressive as a technology demo. It shipped into beta in early 2021.

Then in **July 2021**, NVIDIA renamed Jarvis to **Riva**. Same technology, same team, new name. The rename was clean — all documentation, scripts, and CLI commands were updated. Jarvis Speech Skills became Riva Speech Skills.

But the architecture question remained: Riva handled ASR (speech-to-text) and TTS (text-to-speech) brilliantly. For dialog management — the logic that decides what to say next — you still needed an external framework.

The original Riva sample applications demonstrated two configurations:

- **Config 1**: Riva ASR + Riva TTS + Riva NLP + **Rasa dialog manager**
- **Config 2**: Riva ASR + Riva TTS + **Dialogflow NLU + Dialogflow dialog manager**

Riva did speech. Rasa or Dialogflow did conversation logic. That was the architecture.

---

### The Great Rewrite: LLMs Replace Dialog Management

Then everything changed.

The traditional conversational AI pipeline looked like this:

```
ASR → NLU (intent + entities) → Dialog Manager (state machine) → Fulfillment → NLG → TTS
```

Every platform — Rasa, Dialogflow, Amazon Lex, IBM Watson — required developers to:

- Predefine every **intent** ("book_flight," "check_balance," "reset_password")
- Annotate **entities** (dates, locations, amounts)
- Write **stories and flows** defining conversation paths
- Maintain **thousands of training utterances**

The fundamental problem: **conversations that went off-script broke.** If someone was updating their shipping address and asked about a refund mid-conversation, the system could not handle both. The state machine was rigid.

Large Language Models replaced all of this. Not gradually — fundamentally.

- **No intent schemas required.** LLMs understand user messages without predefined categories.
- **No rigid state machines.** LLMs maintain context and handle digressions naturally.
- **No training utterances.** A system prompt replaces thousands of labeled examples.
- **Dynamic adaptation.** The model weaves new topics into the conversation and returns to the original task.

> **The entire NLU + Dialog Manager pipeline collapsed into a single LLM call.**

This is why NVIDIA's original decision to not build native dialog management looks prescient in hindsight. They bet on speech — the part that LLMs would **not** replace. ASR and TTS are signal processing problems. You need specialized models for those. But dialog management? That is exactly what LLMs are best at.

Rasa recognized this too. Their **CALM** (Conversational AI with Language Models) engine dropped intent classification and entity extraction entirely. The LLM handles "dialogue understanding" natively. Google is integrating Dialogflow CX with Vertex AI agents. The frameworks are adapting because the old approach is obsolete.

---

### Voice Is The Interface That Matters Now

Here is where it gets interesting.

Peter Steinberger — the developer who bootstrapped PSPDFKit into a major company, then created **OpenClaw**, one of the fastest-growing open-source AI agent frameworks — told Lex Fridman something striking:

> **"I don't write, I talk. These hands are too precious for writing now."**

He uses voice input almost exclusively to interact with AI agents. Not typing. Speaking. And he did it so intensely during the development of OpenClaw that **he lost his voice**.

Think about what that means. A developer — someone whose primary interface with computers has been a keyboard for decades — switched to voice because it is the more natural way to interact with an AI agent. And he is not alone.

The voice coding ecosystem is growing:

- **Talon Voice** — scriptable speech recognition tuned for developer commands. Originally adopted by developers with RSI, now expanding to anyone who wants hands-free coding.
- **Cursorless** — a VS Code extension that decorates tokens with colored hats and enables spoken structural code editing. Multiple editing actions in one spoken command.
- **Wispr Flow** — the Whisper-based dictation tool Steinberger calls "king" for voice input. Hit the Fn key, speak, transcribed.

The pattern is clear: **as development shifts toward conversing with AI agents, voice becomes the natural interface.** You do not type to a conversation partner. You speak.

---

### Caterpillar: Voice On The Edge

The most compelling example of all this coming together is Caterpillar.

At **CES 2026**, Caterpillar debuted the **Cat AI Assistant** — a voice-activated system running inside heavy equipment cabs. An operator in a Cat 306 CR Mini Excavator says "Hey Cat, how do I get started?" and gets voice guidance for settings, troubleshooting, and equipment control.

The technical stack:

- **NVIDIA Riva** for speech — Parakeet ASR (speech-to-text) and Magpie TTS (text-to-speech)
- **Qwen3 4B** — a compact LLM for intent parsing and response generation, served locally via vLLM
- **NVIDIA Jetson Thor** — edge AI hardware running everything on-device
- **Caterpillar Helios** — their unified data platform providing equipment context

Notice what is **not** in the stack: Rasa. Dialogflow. Any traditional dialog management framework. The LLM handles conversation flow directly. Riva handles speech. That is the entire pipeline.

And it runs **entirely on the edge**. No cloud. No internet required. On a construction site. In a mine. In places where connectivity is a luxury and latency is unacceptable.

> **This is what the Jarvis architecture was always pointing toward — speech on the edge, intelligence in the model, no dialog management middleware in between.**

---

### What This Architecture Looks Like

```
Operator speaks: "Hey Cat, how do I configure the E-Ceiling?"
                    |
                    v
         +---------------------+
         |   NVIDIA Riva       |
         |   Parakeet ASR      |  <-- Speech-to-text (on-device)
         +----------+----------+
                    |
                    v
         +---------------------+
         |   Qwen3 4B (LLM)   |  <-- Intent + response (on-device)
         |   via vLLM          |      No dialog manager needed
         +----------+----------+
                    |
                    v
         +---------------------+
         |   NVIDIA Riva       |
         |   Magpie TTS        |  <-- Text-to-speech (on-device)
         +----------+----------+
                    |
                    v
         Operator hears: "To configure the E-Ceiling..."
```

Three components. All on the edge. No cloud dependency.

Compare this to the original Jarvis architecture with Rasa or Dialogflow in the middle, plus cloud APIs for NLU. The pipeline has been radically simplified.

---

### What Riva Is Today

Riva has matured significantly since the Jarvis days. Current capabilities (as of late 2025):

**ASR (Speech-to-Text)**
- 12 languages including Arabic, Hindi, Japanese, Korean, Mandarin
- Models: Parakeet CTC 1.1B (English), Parakeet RNNT 1.1B (Multilingual), Whisper Large v3
- Streaming and offline modes, speaker diarization, word-level timestamps
- FP8 quantization for efficiency

**TTS (Text-to-Speech)**
- 7 languages with models including Magpie TTS Multilingual and Magpie TTS Zeroshot
- **Voice cloning from 5 seconds of audio** (Magpie Flow)
- SSML support for prosody control

**Translation**
- Up to 32 languages via neural machine translation
- Text-to-text, speech-to-text, and speech-to-speech translation

**Deployment**
- NVIDIA NIM containers with gRPC, REST, and WebSocket APIs
- Cloud endpoints at `build.nvidia.com` for prototyping
- Edge deployment on Jetson platforms
- Self-hosted with 90-day free trial

The Python client (`nvidia-riva-client`) is at version 2.24.0 and provides high-level wrappers for all of this.

---

### Why This Matters

Three things stand out:

**1. The speech layer is the durable layer.** Dialog management frameworks came and went. Intent schemas came and went. NLU training data came and went. But ASR and TTS? Those are physics problems — converting sound waves to text and back. LLMs did not replace them. They cannot. Riva bet on the right layer.

**2. Voice is the agentic interface.** When you interact with an AI agent, typing is friction. Speaking is natural. Steinberger's experience is not an edge case — it is a leading indicator. As AI agents become the primary way developers (and equipment operators, and customers) interact with software, voice becomes the default input. Riva provides the infrastructure for this.

**3. Edge deployment changes everything.** The Caterpillar example is not a demo. It is production infrastructure running in excavator cabs with no internet. Riva's ability to run ASR and TTS on Jetson hardware, combined with a local LLM, creates fully autonomous voice agents that work anywhere. This is the pattern for industrial AI, automotive AI, and any environment where cloud is not an option.

> **NVIDIA built the speech layer. LLMs replaced the dialog layer. Voice became the interface. And now it all runs on the edge. The original Jarvis vision — just without the parts that turned out to be unnecessary.**

---

### Running The Demo

I built a prototype that demonstrates the Riva pipeline — ASR and TTS via NVIDIA's cloud API. You can record audio, transcribe it with Riva, process it through an LLM, and hear the response spoken back.

```bash
pip install nvidia-riva-client gradio openai
export NVIDIA_API_KEY="your-key"   # Free at build.nvidia.com
export GROK_API_KEY="your-key"     # From x.ai
python3 riva_voice_agent_demo.py
```

The demo opens a web UI where you can speak (or upload audio), see the transcription, and hear the AI agent's response.

---

Follow me on [LinkedIn](https://www.linkedin.com/in/cobusgreyling) for more on Agentic AI, LLMs and NLP.

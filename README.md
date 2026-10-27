# Sous-Chef

A hands-free voice cooking assistant built on [LiveKit Agents](https://github.com/livekit/agents), running 100% locally with open-source models. Tell it what you want to cook; it writes a recipe, reads it to you one step at a time, and sets kitchen timers that announce themselves out loud.

| Piece | Role | Port |
|---|---|---|
| `livekit-server --dev` | Self-hosted WebRTC media server | 7880 |
| `sous-chef/agent.py` | The agent (LiveKit Agents, Python) | — |
| [Speaches](https://github.com/speaches-ai/speaches) (faster-whisper) | Speech-to-text | 8000 |
| [Ollama](https://ollama.com) + `qwen2.5:7b` | LLM with tool calling | 11434 |
| [Kokoro-FastAPI](https://github.com/remsky/Kokoro-FastAPI) | Text-to-speech | 8880 |
| [agent-starter-react](https://github.com/livekit-examples/agent-starter-react) | Browser frontend | 3000 |

## How it works

Two agents share one session:

1. **RecipePicker** asks what you'd like to make. Its `choose_recipe` tool has the LLM write the recipe once (JSON), stores it in session `userdata`, and hands off to…
2. **SousChef**, which only reads steps through tools (`next_step`, `repeat_step`, `previous_step`) so it can't drift from the recipe, and runs timers (`set_timer`, `list_timers`, `cancel_timer`) as asyncio tasks that speak on their own with `session.say()`.

Turn detection and interruptions use local Silero VAD, so nothing calls LiveKit Cloud.

## Setup (macOS)

```bash
brew install uv portaudio livekit livekit-cli ollama
brew install --cask docker   # then open Docker Desktop
npm install -g pnpm
```

### Model servers

```bash
brew services start ollama
ollama pull qwen2.5:7b

docker run -d --name kokoro -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-cpu:latest
docker run -d --name speaches -p 8000:8000 \
  -v hf-hub-cache:/home/ubuntu/.cache/huggingface/hub \
  ghcr.io/speaches-ai/speaches:latest-cpu
curl -X POST http://localhost:8000/v1/models/Systran/faster-distil-whisper-small.en
```

### Agent

```bash
cd sous-chef
cp .env.example .env
uv sync
uv run agent.py download-files
uv run smoke_test.py          # checks STT/LLM/TTS without a mic
uv run script_check.py        # text-mode run of a cooking script
```

### Frontend

```bash
git clone https://github.com/livekit-examples/agent-starter-react.git
cd agent-starter-react
cat > .env.local <<'EOF'
LIVEKIT_TOKEN_SERVER_ID=
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=devkey
LIVEKIT_API_SECRET=secret
AGENT_NAME=
EOF
pnpm install
```

## Run

Use one terminal each:

```bash
livekit-server --dev                                  # 1. media server
cd sous-chef && lk agent dev --dev agent.py           # 2. agent (hot-reloads on save)
cd agent-starter-react && pnpm dev                    # 3. frontend
```

Open http://localhost:3000, allow the microphone (use headphones), and say what you'd like to cook.
Try `uv run agent.py console` in `sous-chef/` to talk to the agent from the terminal without the server or frontend.

> Recipes are written by a small local model. Double-check quantities and cooking times.

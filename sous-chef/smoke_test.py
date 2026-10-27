"""Phase 3 smoke test: drive the agent's own STT/LLM/TTS/VAD objects without a mic.

Run with:  uv run smoke_test.py
"""

import asyncio
import time

from livekit.agents import llm as lk_llm

from agent import build_session


async def main() -> None:
    session = build_session()
    print("VAD   :", type(session.vad).__name__, "loaded")

    # LLM: Ollama through the OpenAI-compatible API
    t = time.perf_counter()
    ctx = lk_llm.ChatContext()
    ctx.add_message(role="user", content="In one short sentence, what can I make with eggs and spinach?")
    reply = ""
    async with session.llm.chat(chat_ctx=ctx) as stream:
        async for chunk in stream:
            if chunk.delta and chunk.delta.content:
                reply += chunk.delta.content
    print(f"LLM   : {reply.strip()!r}  ({time.perf_counter() - t:.2f}s)")

    # TTS: Kokoro turns the reply into audio frames
    t = time.perf_counter()
    frames = []
    async with session.tts.synthesize(reply) as stream:
        async for audio in stream:
            frames.append(audio.frame)
    seconds = sum(f.samples_per_channel / f.sample_rate for f in frames)
    print(f"TTS   : {seconds:.1f}s of audio at {frames[0].sample_rate} Hz  ({time.perf_counter() - t:.2f}s)")

    # STT: Speaches transcribes that audio back to text
    t = time.perf_counter()
    event = await session.stt.recognize(buffer=frames)
    print(f"STT   : {event.alternatives[0].text.strip()!r}  ({time.perf_counter() - t:.2f}s)")

    print("\nAll pipeline pieces work. Now try: uv run agent.py console")


if __name__ == "__main__":
    asyncio.run(main())

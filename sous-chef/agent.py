import asyncio
import json
import logging
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    TurnHandlingOptions,
    cli,
    function_tool,
)
from livekit.plugins import openai, silero
from openai import AsyncOpenAI

load_dotenv()
logger = logging.getLogger("sous-chef")

STT_URL = "http://localhost:8000/v1"  # Speaches
LLM_URL = "http://localhost:11434/v1"  # Ollama
TTS_URL = "http://localhost:8880/v1"  # Kokoro-FastAPI
LLM_MODEL = "qwen2.5:7b"


# ------------------------------------------------------------------ state
@dataclass
class Timer:
    task: asyncio.Task
    ends_at: float  # a time.monotonic() value


@dataclass
class KitchenState:
    recipe_name: str = ""  # empty until the cook picks a dish
    steps: list[str] = field(default_factory=list)
    step: int = -1  # -1 means we haven't started
    timers: dict[str, Timer] = field(default_factory=dict)


# ------------------------------------------------------------------ recipe writer
RECIPE_PROMPT = """Write a simple home-cooking recipe for: {dish}

Reply with JSON only, in exactly this shape:
{{"name": "<short dish name>", "steps": ["<step 1>", "<step 2>", ...]}}

Rules for the steps:
- 5 to 8 steps, in cooking order, with no numbering.
- Each step is one or two short sentences that will be read aloud to a cook whose hands are busy.
- Include quantities and, whenever something cooks, bakes or rests, the time in minutes.
- Write units as full words (tablespoon, teaspoon, grams), never abbreviations like tbsp or g.
- No lists, emojis or markdown inside a step.
"""


async def write_recipe(dish: str) -> tuple[str, list[str]]:
    """Ask the local LLM for a recipe once, so later steps come from state, not memory."""
    client = AsyncOpenAI(base_url=LLM_URL, api_key="not-needed")
    resp = await client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[{"role": "user", "content": RECIPE_PROMPT.format(dish=dish)}],
    )
    data = json.loads(resp.choices[0].message.content or "{}")
    name = str(data.get("name") or dish).strip()
    steps = [str(s).strip() for s in data.get("steps", []) if str(s).strip()]
    if not 3 <= len(steps) <= 12:
        raise ValueError(f"recipe for {dish!r} had {len(steps)} steps")
    return name, steps


# ------------------------------------------------------------------ agent 1: picker
PICKER_INSTRUCTIONS = """
You are Sous-Chef, a hands-free cooking assistant. The cook's hands are busy,
and your replies are spoken aloud: one or two short sentences, never lists,
emojis or markdown.

Your only job right now is to find out what the cook wants to make.
- If the cook names a specific dish, use the choose_recipe tool with that dish.
- If the cook only gives an ingredient or a vague idea, like "something with chicken",
  do not use any tool yet: suggest two specific dishes by name and ask which one they want.
- Never describe ingredients or steps yourself.
- Invoke tools with a real function call. Never write a tool's name in your reply.
"""


class RecipePicker(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=PICKER_INSTRUCTIONS)

    async def on_enter(self) -> None:
        self.session.generate_reply(
            instructions="Say hi as Sous-Chef and ask what the cook would like to make today."
        )

    @function_tool
    async def choose_recipe(self, context: RunContext["KitchenState"], dish: str):
        """Create the recipe for one specific, named dish and start cooking it.

        Only use this once the cook has named a specific dish, like "butter chicken"
        or "pancakes". If they only gave an ingredient or a vague idea, like
        "something with chicken", do not use this tool: suggest two dishes instead.

        Args:
            dish: The specific dish the cook chose, like "butter chicken".
        """
        # Writing a recipe takes a few seconds on a laptop, so fill the silence.
        async with context.with_filler("Great choice, give me a moment to put the recipe together."):
            try:
                name, steps = await write_recipe(dish)
            except Exception:
                logger.exception("could not write a recipe for %r", dish)
                return f"Sorry, I couldn't put together a recipe for {dish}. Ask the cook to pick another dish."
        state = context.userdata
        state.recipe_name, state.steps, state.step = name, steps, -1
        logger.info("recipe ready: %s (%d steps)", name, len(steps))
        # Returning an Agent hands the conversation over to it.
        return SousChef(name)


# ------------------------------------------------------------------ agent 2: sous-chef
SOUS_CHEF_INSTRUCTIONS = """
You are Sous-Chef, a hands-free cooking assistant. The cook's hands are busy,
and your replies are spoken aloud: one or two short sentences, never lists,
emojis or markdown.

Today's recipe is {recipe_name}. You do NOT know its steps; they are only
available through your tools.

How to use tools: invoke them with a real function call. Never write a tool's
name in your reply, never announce what you are about to do, and never describe
a recipe step you did not just receive from a tool.
- The cook is ready, or says next, done or what now: use the next_step tool.
- The cook says again, repeat, or what was that: use the repeat_step tool.
- The cook wants to go back: use the previous_step tool.
- After a tool returns a step, say that step naturally in your own words.
- If a step mentions a cooking time, offer a timer. Use the set_timer tool only after the cook agrees.
- "How long is left": use the list_timers tool. Stopping a timer: use the cancel_timer tool.
"""


class SousChef(Agent):
    def __init__(self, recipe_name: str) -> None:
        super().__init__(instructions=SOUS_CHEF_INSTRUCTIONS.format(recipe_name=recipe_name))
        self.recipe_name = recipe_name

    async def on_enter(self) -> None:
        # Runs when the picker hands over to this agent
        steps = len(self.session.userdata.steps)
        self.session.generate_reply(
            instructions=f"Tell the cook the recipe for {self.recipe_name} is ready, "
            f"that it has {steps} steps, and ask if they're ready for step one."
        )

    # ---- recipe navigation ----
    @function_tool
    async def next_step(self, context: RunContext[KitchenState]) -> str:
        """Move to the next step of the recipe and return its text."""
        state = context.userdata
        if state.step + 1 >= len(state.steps):
            return "There are no more steps. The dish is finished, congratulate the cook."
        state.step += 1
        return self._describe(state)

    @function_tool
    async def repeat_step(self, context: RunContext[KitchenState]) -> str:
        """Repeat the current recipe step."""
        state = context.userdata
        if state.step < 0:
            return "We haven't started yet. Ask if the cook is ready for step one."
        return self._describe(state)

    @function_tool
    async def previous_step(self, context: RunContext[KitchenState]) -> str:
        """Go back one step and return its text."""
        state = context.userdata
        state.step = max(0, state.step - 1)
        return self._describe(state)

    def _describe(self, state: KitchenState) -> str:
        return f"Step {state.step + 1} of {len(state.steps)}: {state.steps[state.step]}"

    # ---- timers ----
    @function_tool
    async def set_timer(
        self, context: RunContext[KitchenState], label: str, minutes: float
    ) -> str:
        """Start a kitchen timer that announces itself out loud when it finishes.

        Args:
            label: The food being timed, taken from the current recipe step, like "pasta" or "garlic".
                Never use words from the cook's sentence such as "test".
            minutes: Length of the timer in minutes. Decimals are fine, e.g. 0.5.
        """
        state = context.userdata
        label = label.strip().lower()
        if label in state.timers:
            return f"A {label} timer is already running."
        seconds = max(1.0, minutes * 60)
        task = asyncio.create_task(self._ring_later(state, label, seconds))
        state.timers[label] = Timer(task=task, ends_at=time.monotonic() + seconds)
        return f"The {label} timer is set for {minutes:g} minutes."

    @function_tool
    async def list_timers(self, context: RunContext[KitchenState]) -> str:
        """Say which timers are running and how much time each has left."""
        state = context.userdata
        if not state.timers:
            return "No timers are running."
        now = time.monotonic()
        parts = []
        for label, t in state.timers.items():
            left = max(0, int(t.ends_at - now))
            parts.append(f"{label}: {left // 60} minutes {left % 60} seconds left")
        return "; ".join(parts)

    @function_tool
    async def cancel_timer(self, context: RunContext[KitchenState], label: str) -> str:
        """Cancel a running timer.

        Args:
            label: Name of the timer to cancel.
        """
        state = context.userdata
        timer = state.timers.pop(label.strip().lower(), None)
        if timer is None:
            return f"There is no {label} timer running."
        timer.task.cancel()
        return f"The {label} timer is cancelled."

    async def _ring_later(self, state: KitchenState, label: str, seconds: float) -> None:
        await asyncio.sleep(seconds)  # cancel_timer interrupts this sleep
        state.timers.pop(label, None)
        # Agent-initiated speech: nobody asked, the agent just talks.
        self.session.say(f"Ding ding! Your {label} timer is done.")


# ------------------------------------------------------------------ session
def build_session(**kwargs) -> AgentSession:
    return AgentSession(
        vad=silero.VAD.load(),
        stt=openai.STT(
            model="Systran/faster-distil-whisper-small.en",
            base_url=STT_URL,
            api_key="not-needed",
            language="en",
        ),
        llm=openai.LLM.with_ollama(model=LLM_MODEL, base_url=LLM_URL, temperature=0.3),
        tts=openai.TTS(
            model="kokoro",
            voice="af_alloy",
            base_url=TTS_URL,
            api_key="not-needed",
            response_format="wav",
        ),
        turn_handling=TurnHandlingOptions(
            turn_detection="vad",
            interruption={"mode": "vad"},
        ),
        **kwargs,
    )


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    state = KitchenState()
    session = build_session(userdata=state)

    async def cancel_all_timers() -> None:
        for t in state.timers.values():
            t.task.cancel()

    # Clean up background timers when the call ends
    ctx.add_shutdown_callback(cancel_all_timers)

    await session.start(agent=RecipePicker(), room=ctx.room)


if __name__ == "__main__":
    cli.run_app(server)

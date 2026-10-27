"""Phase 5 check: run a cooking script in text mode (real LLM + tools, no audio).

Run with:  uv run script_check.py ["dish to ask for"]
"""

import asyncio
import sys

from livekit.agents import AgentSession
from livekit.plugins import openai

from agent import LLM_MODEL, LLM_URL, KitchenState, RecipePicker

DISH = sys.argv[1] if len(sys.argv) > 1 else "garlic butter spaghetti"

SCRIPT = [
    (f"I'd like to make {DISH}.", "choose_recipe"),
    ("I'm ready.", "next_step"),
    ("Done, what's next?", "next_step"),
    ("Set a timer for half a minute so I can test.", "set_timer"),
    ("How long is left on the timer?", "list_timers"),
    ("Repeat the step.", "repeat_step"),
    ("Go back one step.", "previous_step"),
]


def describe(ev) -> str:
    item = ev.item
    if ev.type == "function_call":
        return f"  -> tool call   {item.name}({item.arguments})"
    if ev.type == "function_call_output":
        return f"  <- tool result {item.output!r}"
    if ev.type == "message":
        return f"  agent says     {item.text_content!r}"
    if ev.type == "agent_handoff":
        return f"  == handoff     {item.old_agent_id} -> {item.new_agent_id}"
    return f"  {ev.type}"


async def main() -> None:
    state = KitchenState()
    llm = openai.LLM.with_ollama(model=LLM_MODEL, base_url=LLM_URL, temperature=0.3)
    passed = 0
    async with llm, AgentSession(llm=llm, userdata=state) as session:
        await session.start(RecipePicker())
        for text, expected in SCRIPT:
            result = await session.run(user_input=text)
            called = [ev.item.name for ev in result.events if ev.type == "function_call"]
            ok = expected in called
            passed += ok
            print(f"\nCOOK: {text!r}   (expect {expected})  {'PASS' if ok else 'FAIL'}")
            for ev in result.events:
                print(describe(ev))
            if expected == "choose_recipe" and state.steps:
                print(f"  recipe: {state.recipe_name}")
                for i, s in enumerate(state.steps, 1):
                    print(f"    {i}. {s}")
            print(f"  state: step={state.step}, timers={list(state.timers)}")
    print(f"\n{passed}/{len(SCRIPT)} turns called the expected tool")


if __name__ == "__main__":
    asyncio.run(main())

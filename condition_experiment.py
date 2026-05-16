"""Run a single experiment in one of Riedl's three conditions (Plain / Persona / ToM).

Riedl's released `persona_experiment.py` uses a generic prompt that does not
match any of the three App. A.1 prompt boxes exactly, and has no condition
selector. This file:

  1. Implements all three conditions using prompt text matching Riedl's
     App. A.1 verbatim.
  2. Exposes a clean `condition` parameter so we can run any of them.
  3. Leaves Riedl's original code untouched (we just import GameMaster / Agent
     from it).

Usage:
    python condition_experiment.py plain
    python condition_experiment.py persona
    python condition_experiment.py tom

Defaults: gpt-4o-mini, N=10, T=1.0, sum mode, 20-round cap, 1 run.
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import time
from typing import List, Optional

from experiment import GameMaster, Agent, Round
from llm_run import chat
from persona_wrapper import PersonaWrapper
from prompt_capture import capture_prompt


# ---- Prompt fragments matching Riedl App. A.1 verbatim ----

# Common preamble (used by all three conditions)
_PREAMBLE = (
    "You are playing a sum guessing game. Your goal is to help your group "
    "sum to the mystery number. Your guess range is 0 to 50."
)

# Strategy instruction shared by Plain / Persona / ToM
_STRATEGY = (
    "What is your guess this round? Always start with the efficient strategy "
    "in guessing games which is to use a binary search approach: guessing the "
    "midpoint of the current range. Always anchor your guess on the group "
    "feedback from previous rounds (too high / too low)."
)

# Extra ToM instruction appended only in the ToM condition
_TOM_EXTRA = (
    " Only as a secondary approach, carefully think through step-by-step what "
    "others might guess and how the contributions of others contribute to the "
    "sum of the group guesses for the mystery number. Consider what roles "
    "other agents might be playing (e.g., guessing higher or lower) and adapt "
    "your own adjustment to complement the group."
)

_FOOTER = "End your answer with: FINAL GUESS: [0-50]"


def build_prompt(
    condition: str,
    game_history: List[Round],
    agent_id: int,
    persona_text: Optional[str],
    round_num: int,
) -> str:
    """Build the prompt for a given condition, agent, and round.

    condition: 'plain' | 'persona' | 'tom'
    persona_text: None for plain; the persona paragraph for persona/tom.
    """
    # Game history block
    if round_num == 1:
        history_block = (
            "Game History:\n"
            "    This is the first round and you have not made any guesses yet.\n"
        )
    else:
        history_block = "Game History:\n"
        for round_data in game_history:
            history_block += f"Round {round_data.round_num}: "
            history_block += f"Your guess: {round_data.guesses[agent_id]}\n"
            history_block += f"Result: {round_data.feedback}\n"

    # Strategy section: ToM appends extra reasoning instruction
    strategy_block = _STRATEGY + (_TOM_EXTRA if condition == "tom" else "")

    body = f"{_PREAMBLE}\n\n{history_block}\n{strategy_block}\n\n{_FOOTER}"

    if condition == "plain":
        return body
    elif condition in ("persona", "tom"):
        if persona_text is None:
            raise ValueError(f"persona_text required for condition='{condition}'")
        # Riedl A.1: persona text precedes the body, separated by a blank line.
        return f"{persona_text}\n\n{body}"
    else:
        raise ValueError(f"Unknown condition: {condition!r}. Use plain/persona/tom.")


# ---- Agent that uses condition-aware prompting ----

class ConditionAgent(Agent):
    def __init__(
        self,
        agent_id: int,
        model: str,
        temperature: float,
        condition: str,
        persona_text: Optional[str],
    ):
        super().__init__(agent_id, model, temperature)
        self.condition = condition
        self.persona_text = persona_text

    # Class-level container so we can save responses across all agents/rounds.
    _response_log: list = []

    async def make_guess(self, round_num, game_history, guess_range, mode):
        prompt = build_prompt(
            condition=self.condition,
            game_history=game_history,
            agent_id=self.agent_id,
            persona_text=self.persona_text,
            round_num=round_num,
        )
        capture_prompt(round_num, self.agent_id, prompt)

        response = await chat(
            model=self.model,
            prompt=prompt,
            temperature=self.temperature,
            max_tokens=300,  # generous for ToM reasoning traces
        )

        # Save the full response text so we can read reasoning traces later.
        ConditionAgent._response_log.append({
            "round": round_num,
            "agent_id": self.agent_id,
            "condition": self.condition,
            "response": response,
        })

        # Try to extract "FINAL GUESS: NN" first
        content = response if isinstance(response, str) else str(response)
        m = re.search(r"FINAL GUESS:\s*(\d+)", content, re.IGNORECASE)
        if m:
            guess = int(m.group(1))
            if guess_range[0] <= guess <= guess_range[1]:
                self.guess_history.append(guess)
                self.last_successful_guess = guess
                return guess, prompt, response, False

        # Fall back to base-class robust extraction
        try:
            guess = self._extract_number_robust(response, guess_range)
            self.guess_history.append(guess)
            self.last_successful_guess = guess
            return guess, prompt, response, False
        except Exception:
            fallback = self.last_successful_guess or ((guess_range[0] + guess_range[1]) // 2)
            self.guess_history.append(fallback)
            return fallback, prompt, response, True


# ---- GameMaster that wires in personas and the condition ----

class ConditionGameMaster(GameMaster):
    def __init__(
        self,
        condition: str,
        persona_wrapper: Optional[PersonaWrapper] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.condition = condition
        self.persona_wrapper = persona_wrapper

    def add_agent(self, model: str) -> ConditionAgent:
        agent_id = len(self.agents)
        persona_text = None
        if self.condition in ("persona", "tom"):
            if self.persona_wrapper is None:
                raise RuntimeError(
                    f"condition={self.condition!r} requires a PersonaWrapper"
                )
            persona_text = self.persona_wrapper.agent_personas.get(agent_id)
        agent = ConditionAgent(
            agent_id=agent_id,
            model=model,
            temperature=self.temperature,
            condition=self.condition,
            persona_text=persona_text,
        )
        self.agents.append(agent)
        return agent


# ---- Single-experiment runner ----

async def run_one(
    condition: str,
    model: str = "gpt-4o-mini",
    num_agents: int = 10,
    temperature: float = 1.0,
    max_rounds: int = 20,
    persona_file: str = "personas_gpt41.txt",
) -> str:
    """Run a single experiment in the given condition. Returns the results dir."""
    assert condition in ("plain", "persona", "tom"), condition

    persona_wrapper = None
    if condition in ("persona", "tom"):
        persona_wrapper = PersonaWrapper(persona_file)
        persona_wrapper.assign_personas(num_agents)

    game = ConditionGameMaster(
        condition=condition,
        persona_wrapper=persona_wrapper,
        mode="sum",
        temperature=temperature,
        max_rounds=max_rounds,
        num_agents=num_agents,
        run_id=1,
    )

    for _ in range(num_agents):
        game.add_agent(model)

    # Reset response log at the start of each run.
    ConditionAgent._response_log = []

    start = time.time()
    await game.play_game()
    elapsed = time.time() - start

    # Persist the response log next to the rest of the run's outputs.
    import json as _json
    log_path = os.path.join(game.results_dir, "agent_responses.json")
    with open(log_path, "w", encoding="utf-8") as f:
        _json.dump(ConditionAgent._response_log, f, indent=2, ensure_ascii=False)

    print(f"\n[{condition}] results in: {game.results_dir}")
    print(f"[{condition}] elapsed: {elapsed:.1f}s")
    print(f"[{condition}] reasoning traces: {log_path}")
    return game.results_dir


# ---- CLI ----

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("plain", "persona", "tom"):
        print("Usage: python condition_experiment.py {plain|persona|tom} [model]")
        print("       (model defaults to gpt-4o-mini)")
        sys.exit(1)
    condition = sys.argv[1]
    model = sys.argv[2] if len(sys.argv) >= 3 else "gpt-4o-mini"
    print(f"Running condition={condition} model={model}")
    asyncio.run(run_one(condition=condition, model=model))

"""The coding tasks: narrow visible tests that the reward pays, broader hidden tests that label the result.

The task definitions live in ``safeinterp/rlhack/env.py`` (``TASKS``): 30 training
tasks plus 8 held out for ``rlhack-monitor``.  Each prompt shows ``N_VISIBLE``
asserts.  The reward only runs those, so special-casing them is the cheapest way
to score 1.0.  Ten hidden tests per task are never rewarded and only label a
completion as honest or hacked.

To make hacking pay more (fix #2 for the gate in step 3), lower ``N_VISIBLE``:
fewer visible inputs means hardcoding them is even cheaper than solving the task.

    python reward_hacking/tasks.py                 # list the training tasks and their visible tests
"""
from __future__ import annotations

from safeinterp.rlhack.env import HINTS, SYSTEM, TASKS, Task, asserts, build_tasks, prompt  # noqa: F401

N_VISIBLE = 3  # asserts shown in the prompt and paid by the reward
N_HIDDEN = 10  # held-out tests, never rewarded


def train_tasks(n_visible: int = N_VISIBLE, seed: int = 0) -> list[Task]:
    """The 30 RL training tasks with their visible and hidden tests filled in."""
    return build_tasks(n_visible=n_visible, n_hidden=N_HIDDEN, seed=seed, split="train")


if __name__ == "__main__":
    ts = train_tasks()
    print(f"{len(ts)} training tasks, {N_VISIBLE} visible / {N_HIDDEN} hidden tests each\n")
    for t in ts:
        print(f"{t.name:22s} {t.signature}")
        print("    " + asserts(t).replace("\n", "\n    "))

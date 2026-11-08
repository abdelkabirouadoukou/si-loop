"""study2: the reference (non-synthetic) Study-2 implementation.

    tasks.py       local problem suite (string + algorithmic + a held-out parsing family) with oracles
    seed_repo.py   the cycle-0 agent (a library of task functions, some buggy / missing)
    verifier.py    PytestVerifier: hidden pytest cases, candidate code isolated in a sandbox process
    llm_meta.py    LLMMetaAgent + clients (OpenAI-compatible / Anthropic / offline demo)
run_study2.py (repo root) wires them into the unchanged kernel.
"""

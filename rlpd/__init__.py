"""RLPD reproduction package.

Everyone imports the shared types from `rlpd.interfaces`; nobody redefines the
batch contract or the agent/buffer signatures locally. Real implementations are
swapped in behind those signatures so the three workstreams never block on each
other:

    networks.py, sac.py          algorithm   (Member 1)
    replay_buffer.py, dataset.py,
    envs.py, evaluate.py         data/env/eval (Member 2)
    ../train.py, ../run.py       wiring/infra  (Member 3)
"""

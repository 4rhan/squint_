"""Time QC-FQL gradient updates on the GPU: old path (one sample + one compiled update per step) vs the
new path (batches sampled in bulk + compiled + CUDA-graph captured update). Synthetic buffers, no env.

    python -m examples.bench_qc_update                       # tp3_recovery sizes (512 x 4, batch 512)
    python -m examples.bench_qc_update --hidden_dim 256 --num_layers 3

Prints gradient steps/s for an online block (critic every step, actor every 4th) and for offline
pretraining (actor every step), and the projected time of 150k offline + 1.5M online (1024 envs) steps.
"""
import argparse
import time

import torch

import train_squint  # noqa: F401  (sets TF32 like the training scripts)
from qc_agent import QCAgent, QCConfig
from qc_data import ChunkBuffer

p = argparse.ArgumentParser()
p.add_argument("--hidden_dim", type=int, default=512)
p.add_argument("--num_layers", type=int, default=4)
p.add_argument("--batch_size", type=int, default=512)
p.add_argument("--num_updates", type=int, default=256)
p.add_argument("--policy_frequency", type=int, default=4)
p.add_argument("--image_size", type=int, default=16)
p.add_argument("--n_state", type=int, default=12)
p.add_argument("--n_act", type=int, default=6)
p.add_argument("--horizon", type=int, default=5)
p.add_argument("--blocks", type=int, default=6, help="timed blocks of num_updates steps per mode")
p.add_argument("--modes", default="eager,compile,compile+graphs")
a = p.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_obs = (a.image_size, a.image_size, 3)


def make_buffer(T, E):
    rb = ChunkBuffer(T, E, n_obs, a.n_state, a.n_act, dev)
    rb.extend(torch.randint(0, 255, (T, E, *n_obs), dtype=torch.uint8, device=dev),
              torch.randn(T, E, a.n_state, device=dev),
              torch.randint(0, 255, (T, E, *n_obs), dtype=torch.uint8, device=dev),
              torch.randn(T, E, a.n_state, device=dev), torch.rand(T, E, a.n_act, device=dev) * 2 - 1,
              torch.randn(T, E, device=dev), torch.zeros(T, E, device=dev),
              (torch.rand(T, E, device=dev) < 1 / 300).float())
    return rb


online, demos = make_buffer(300, 1024), make_buffer(90_000, 1)
B, h = a.batch_size, a.horizon
n_on = n_off = B // 2


def one_batch():
    parts = [online.sample(n_on, h, 0.99), demos.sample(n_off, h, 0.99)]
    return {k: torch.cat([q[k] for q in parts], 0) for k in parts[0]}


def bulk(k):
    parts = [{key: v.view(k, n, *v.shape[1:]) for key, v in buf.sample(k * n, h, 0.99).items()}
             for buf, n in ((online, n_on), (demos, n_off))]
    block = {key: torch.cat([q[key] for q in parts], 1) for key in parts[0]}
    return [{key: v[i] for key, v in block.items()} for i in range(k)]


def run_block(agent, use_bulk, actor_every):
    batches = bulk(a.num_updates) if use_bulk else None
    for i in range(a.num_updates):
        agent.update(batches[i] if use_bulk else one_batch(), update_actor=i % actor_every == 0)


results = {}
for mode in a.modes.split(","):
    cfg = QCConfig(horizon=h, hidden_dim=a.hidden_dim, actor_layers=a.num_layers, critic_hidden_dim=a.hidden_dim,
                   critic_layers=a.num_layers, gamma=0.99)
    agent = QCAgent(cfg, n_obs, a.n_state, a.n_act, dev, compile="compile" in mode, cudagraphs="graphs" in mode)
    use_bulk = "graphs" in mode
    for name, every in (("online", a.policy_frequency), ("offline", 1)):
        t0 = time.perf_counter()
        run_block(agent, use_bulk, every)  # warmup / compile / capture
        torch.cuda.synchronize() if dev.type == "cuda" else None
        warm = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(a.blocks):
            run_block(agent, use_bulk, every)
        torch.cuda.synchronize() if dev.type == "cuda" else None
        sps = a.blocks * a.num_updates / (time.perf_counter() - t0)
        results[(mode, name)] = sps
        print(f"{mode:16s} {name:8s} {sps:7.1f} grad steps/s   (first block incl. compile: {warm:.1f}s)", flush=True)
    del agent
    torch.cuda.empty_cache()

print("\nprojected update time for 150k offline + 1.5M online steps (1024 envs, 256 updates/step):")
n_iter = 1_500_000 // 1024
for mode in a.modes.split(","):
    off = 150_000 / results[(mode, "offline")] / 60
    on = n_iter * a.num_updates / results[(mode, "online")] / 60
    print(f"  {mode:16s} offline {off:5.1f} min + online {on:5.1f} min = {off + on:5.1f} min (env stepping and evals not included)")

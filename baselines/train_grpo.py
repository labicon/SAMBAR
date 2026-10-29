r"""Continual RL for pi0.5 on LIBERO: sequential GRPO with whole-model LoRA.

Port of UT-Austin-RobIn/continual-vla-rl ("Simple Recipe Works: VLA Models are Natural Continual
Learners with RL", Hu et al. 2026) — Seq. FT = train task k with online GRPO + LoRA starting from
the task k-1 adapter, no replay / regularizer. One invocation = one sequential stage (one task);
scripts/train_grpo_lora_seq_libero.sh chains 3 -> 8 -> 11 on top of the 32-task pretrain ckpt.

Faithful to the paper's RLinf config (examples/embodiment/config/crl_experiment/
libero_*_grpo_openvlaoft_*.yaml) wherever pi0.5 allows:
  * env: LIBERO, max_episode_steps=512, fixed/ordered init-state ids (shuffled 50-state permutation,
    seed 0, consumed num_group_envs at a time), auto_reset=False, ignore_terminations=False
    (episode ends at first success), reward = 5 · success (relative reward -> +5 once).
  * rollout: rollout_epoch=16 × num_group_envs=8 × group_size=8 = 1024 episodes per epoch;
    max_epochs = 5 (libero_10 suite) / 10 (others).
  * GRPO: score = episode return, A = (score − group mean)/(group std + 1e-6) broadcast to every
    step; groups with mean return outside [0.5, 4.5] (all-fail / all-success) are filtered out.
  * loss: token-level PPO clip (clip_low 0.2, clip_high 0.28), no KL, no entropy bonus;
    per-episode normalization by (valid env steps / max_episode_steps), mean over the global batch
    (padded samples counted, exactly like RLinf's loss_mask handling).
  * optimizer: AdamW(lr 2e-5, betas 0.9/0.999, eps 1e-8, weight_decay 0.01), grad clip 1.0
    (eps/weight_decay = torch AdamW defaults, which is what RLinf actually uses: it passes only
    params/lr/betas, so its config's adam_eps 1e-5 is dead);
    8 optimizer steps per epoch (= 16·64 episodes · 64 chunk-steps / global_batch_size 8192),
    micro_batch 32, one pass over the rollout.
  * LoRA: rank 32, alpha 32, A ~ N(0, 1/r), B = 0 on every linear of the model (config
    libero_grpo_lora: Gemma 2B + 300M attn/ffn, SigLIP attn/mlp/head, action/time projections);
    everything else frozen. Sequential stages CONTINUE the same adapter (paper's lora_path).

pi0.5-specific (the paper only supports autoregressive-token VLAs):
  * pi0.5 is a flow-matching policy with no token log-probs. We use a Flow-SDE policy
    (Flow-GRPO / πRL): each Euler denoising step t -> t-h is made stochastic with the
    marginal-preserving reverse SDE
        x_{t-h} ~ N(x_t − h·v − h·σ_t²/(2t)·(x_t + (1−t)·v),  σ_t²·h),   σ_t = a·sqrt(t/(1−t))
    so log π(chunk) = Σ_k log N(x_{k+1}; μ_θ(x_k, t_k), σ²h) over the K denoising steps.
    `--grpo.noise-level` (a) plays the role of the paper's sampling temperature.
  * "token" = (denoise step, executed action, action dim): logratios are masked to the actions the
    env actually executed (first replan_steps=5 of the 10-step chunk, up to success) and 7 dims.
  * action chunking follows the openpi LIBERO eval (replan every 5 actions, 10 dummy wait steps).

Process layout: this trainer (openpi venv, JAX) + group_size·num_group_envs LIBERO worker processes
(baselines/libero_rl_env_worker.py in examples/libero/.venv), talking over localhost sockets.

Usage (stage 1: task 3 from the 32-task pretrain ckpt):
    uv run python -m baselines.train_grpo \
        --exp-name pi05_grpo_lora_seq_task3_seed1234 --task-id 3 \
        --pretrained-path ./checkpoints/sambar_narrow/pi05_bc_pretrain_32of40_seed7_lr1e-5/15000/params
Stage 2 (task 8, continue the task-3 adapter):
    uv run python -m baselines.train_grpo \
        --exp-name pi05_grpo_lora_seq_task3_to_task8_seed1234 --task-id 8 --old-task-ids 3 \
        --pretrained-path ./checkpoints/libero_grpo_lora/pi05_grpo_lora_seq_task3_seed1234/5/params
"""

from __future__ import annotations

import concurrent.futures as cf
import dataclasses
import functools
import json
import logging
import math
from multiprocessing.connection import Listener
import os
import pathlib
import platform
import secrets
import subprocess
import sys
import time
from typing import Any

import einops
import etils.epath as epath
import flax.nnx as nnx
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import msgpack_numpy
import optax
import tqdm_loggable.auto as tqdm
import tyro
import wandb

import openpi.models.model as _model
import openpi.models.pi0 as _pi0
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders
import openpi.transforms as _transforms

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from openpi.training.task_split import _LIBERO_SUITE_RANGES  # noqa: E402
from openpi.training.task_split import SUITE_TO_BENCHMARK  # noqa: E402
from openpi.training.task_split import sample_pretrain_tasks  # noqa: E402
from openpi.training.task_split import task_strings_for_ids  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
LIBERO_ACTION_DIM = 7


@dataclasses.dataclass(frozen=True)
class GrpoArgs:
    """GRPO / rollout hyperparameters. Defaults = the paper's RLinf LIBERO GRPO config."""

    group_size: int = 8
    num_group_envs: int = 8
    rollout_epoch: int = 16
    # None -> paper per-suite default: 5 for libero_10 (long), 10 otherwise.
    max_epochs: int | None = None
    max_episode_steps: int = 512
    num_steps_wait: int = 10
    # Actions executed per policy query (openpi LIBERO eval replans every 5 of the 10 predicted).
    replan_steps: int = 5
    reward_coef: float = 5.0
    filter_rewards: bool = True
    rewards_lower_bound: float = 0.5
    rewards_upper_bound: float = 4.5
    normalize_advantages: bool = True
    adv_eps: float = 1e-6
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.28

    lr: float = 2e-5
    adam_b1: float = 0.9
    adam_b2: float = 0.999
    # NOTE: the paper's YAML says adam_eps 1e-5, but RLinf never passes eps to torch AdamW
    # (param groups set only params/lr/betas), so its EFFECTIVE eps is torch's default 1e-8.
    # 1e-5 swamps our gradients (~2e-7 per param) and shrinks updates ~50x.
    adam_eps: float = 1e-8
    weight_decay: float = 0.01
    clip_grad: float = 1.0
    # Optimizer steps per epoch. Paper: 1024 episodes · 64 chunk-steps / global_batch_size 8192 = 8.
    updates_per_epoch: int = 8
    micro_batch_size: int = 32

    # Flow-SDE policy (pi0.5-specific; see module docstring).
    num_denoise_steps: int = 10
    noise_level: float = 0.5


@dataclasses.dataclass(frozen=True)
class CliArgs:
    exp_name: str
    # Dataset task_index (0-9 libero_10, 10-19 goal, 20-29 object, 30-39 spatial) learned in this stage.
    task_id: int
    pretrained_path: str

    config_name: str = "libero_grpo_lora"
    # Tasks learned in earlier sequential stages (provenance only; Seq. FT uses no old-task data).
    old_task_ids: tuple[int, ...] = ()
    split_seed: int = 7
    tasks_per_suite: int = 8

    seed: int = 1234
    # GPUs used by the LIBERO workers for EGL rendering (round-robin). Default: CUDA_VISIBLE_DEVICES.
    env_gpus: str | None = None
    libero_venv: str = str(REPO_ROOT / "examples/libero/.venv")
    rpc_port: int = 0  # 0 = pick a free port
    env_connect_timeout_s: float = 600.0

    checkpoint_base_dir: str | None = None
    overwrite: bool = False
    wandb_enabled: bool = False

    grpo: GrpoArgs = dataclasses.field(default_factory=GrpoArgs)


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.basicConfig(level=logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def suite_for_task(task_id: int) -> str:
    for suite, rng in _LIBERO_SUITE_RANGES.items():
        if task_id in rng:
            return SUITE_TO_BENCHMARK[suite]
    raise ValueError(f"task_id {task_id} outside LIBERO-40")


# === train-state init (mirrors the shared training state initialization) ==============


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


def init_train_state(
    config: _config.TrainConfig, init_rng: at.KeyArrayLike, mesh: jax.sharding.Mesh, tx: optax.GradientTransformation
) -> tuple[training_utils.TrainState, Any]:
    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        model = config.model.create(model_rng)
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)
        params = nnx.state(model)
        params = nnx_utils.state_map(params, config.freeze_filter, lambda p: p.replace(p.value.astype(jnp.bfloat16)))
        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=None,
            ema_params=None,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=False)
    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    train_state = jax.jit(
        init,
        donate_argnums=(1,),
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)
    return train_state, state_sharding


# === Flow-SDE policy ========================================================================


def _sde_times(num_steps: int) -> list[float]:
    """t_k for k = 0..K-1, from noise (t=1) toward data (t=0) with step h = 1/K."""
    return [1.0 - k / num_steps for k in range(num_steps)]


def _sde_sigma(t, h: float, noise_level: float):
    t_c = jnp.minimum(t, 1.0 - h)  # σ_t = a·sqrt(t/(1−t)) diverges at t=1; clamp to the next grid point
    return noise_level * jnp.sqrt(t_c / (1.0 - t_c))


def _prefix(model: _pi0.Pi0, obs: _model.Observation):
    obs = _model.preprocess_observation(None, obs, train=False)
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(obs)
    prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
    return obs, prefix_mask, kv_cache


def _velocity(model: _pi0.Pi0, obs, prefix_mask, kv_cache, x_t, t: float):
    """v_θ(x_t, t) with the prefix KV cache (same computation as Pi0.sample_actions' step)."""
    b = x_t.shape[0]
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(
        obs, x_t, jnp.broadcast_to(jnp.asarray(t, jnp.float32), (b,))
    )
    suffix_attn_mask = _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
    prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
    full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
    positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
    (_, suffix_out), _ = model.PaliGemma.llm(
        [None, suffix_tokens],
        mask=full_attn_mask,
        positions=positions,
        kv_cache=kv_cache,
        adarms_cond=[None, adarms_cond],
    )
    return model._proj("action_out_proj", suffix_out[:, -model.action_horizon :])  # noqa: SLF001


def _sde_mean_std(x_t, v_t, t: float, h: float, noise_level: float):
    x_t = x_t.astype(jnp.float32)
    v_t = v_t.astype(jnp.float32)
    sigma = _sde_sigma(jnp.float32(t), h, noise_level)
    mean = x_t - h * v_t - h * sigma**2 / (2.0 * t) * (x_t + (1.0 - t) * v_t)
    return mean, sigma * math.sqrt(h)


def sample_chain(gargs: GrpoArgs, rng, state: training_utils.TrainState, obs: _model.Observation):
    """Roll out the stochastic denoising chain. Returns x_0..x_K, shape [b, K+1, H, D] (x_K = action)."""
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    K = gargs.num_denoise_steps
    h = 1.0 / K
    obs, prefix_mask, kv_cache = _prefix(model, obs)
    b = obs.state.shape[0]
    x = jax.random.normal(jax.random.fold_in(rng, K), (b, model.action_horizon, model.action_dim))
    chain = [x]
    for k, t in enumerate(_sde_times(K)):
        v = _velocity(model, obs, prefix_mask, kv_cache, x, t)
        mean, std = _sde_mean_std(x, v, t, h, gargs.noise_level)
        x = mean + std * jax.random.normal(jax.random.fold_in(rng, k), mean.shape)
        chain.append(x)
    return jnp.stack(chain, axis=1)


def chain_logprob(gargs: GrpoArgs, model: _pi0.Pi0, obs: _model.Observation, chain):
    """Per-element log N(x_{k+1}; μ_θ(x_k,t_k), σ²h) over the executed actions / real action dims.
    Returns [b, K, replan_steps, 7]."""
    K = gargs.num_denoise_steps
    h = 1.0 / K
    R = gargs.replan_steps
    obs, prefix_mask, kv_cache = _prefix(model, obs)
    out = []
    for k, t in enumerate(_sde_times(K)):
        x_t, x_next = chain[:, k], chain[:, k + 1]
        v = _velocity(model, obs, prefix_mask, kv_cache, x_t, t)
        mean, std = _sde_mean_std(x_t, v, t, h, gargs.noise_level)
        z = (x_next.astype(jnp.float32) - mean) / std
        lp = -0.5 * z**2 - jnp.log(std) - 0.5 * jnp.log(2.0 * jnp.pi)
        out.append(lp[:, :R, :LIBERO_ACTION_DIM])
    return jnp.stack(out, axis=1)


# === GRPO loss ==============================================================================


def old_logprob_step(gargs: GrpoArgs, state: training_utils.TrainState, batch: dict):
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    return chain_logprob(gargs, model, _model.Observation.from_dict(batch["obs"]), batch["chain"])


def grad_step(gargs: GrpoArgs, config: _config.TrainConfig, state: training_utils.TrainState, batch: dict):
    """Gradient of the (partial) global-batch loss on one micro-batch.

    batch: obs, chain [b,K+1,H,D], old_logp [b,K,R,7], adv [b], act_mask [b,R], weight [b]
           (= max_episode_steps / valid_env_steps_of_episode, 0 for pad rows), normalizer [] (global
           tokens per update incl. padded samples). Summing micro-batch losses = RLinf's loss.
    """
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    obs = _model.Observation.from_dict(batch["obs"])

    def loss_fn(m):
        logp = chain_logprob(gargs, m, obs, batch["chain"])
        logratio = logp - batch["old_logp"]
        ratio = jnp.exp(logratio)
        adv = batch["adv"][:, None, None, None]
        pl1 = -adv * ratio
        pl2 = -adv * jnp.clip(ratio, 1.0 - gargs.clip_ratio_low, 1.0 + gargs.clip_ratio_high)
        mask = jnp.broadcast_to(batch["act_mask"][:, None, :, None], logp.shape).astype(jnp.float32)
        w = batch["weight"][:, None, None, None] * mask
        loss = jnp.sum(jnp.maximum(pl1, pl2) * w) / batch["normalizer"]
        n = jnp.sum(mask * (batch["weight"][:, None, None, None] > 0))
        aux = {
            "n_tokens": n,
            "clipfrac_sum": jnp.sum((pl2 > pl1) * mask),
            "approx_kl_sum": jnp.sum(-logratio * mask),
            "ratio_sum": jnp.sum(ratio * mask),
        }
        return loss, aux

    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, aux), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(model)
    return grads, {"loss": loss, **aux}


def apply_step(config: _config.TrainConfig, state: training_utils.TrainState, grads):
    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)
    model = nnx.merge(state.model_def, state.params)
    nnx.update(model, new_params)
    new_state = dataclasses.replace(state, step=state.step + 1, params=nnx.state(model), opt_state=new_opt_state)
    return new_state, optax.global_norm(grads)


# === LIBERO workers ===========================================================================


class EnvPool:
    """group_size·num_group_envs LIBERO worker processes, driven in parallel over sockets."""

    def __init__(self, args: CliArgs, n_envs: int, suite: str, language: str):
        authkey = secrets.token_hex(16)
        # backlog must cover every worker: with the default backlog=1, simultaneous connects overflow the
        # kernel accept queue and some workers hang forever in the auth handshake (seen with 64 envs).
        self.listener = Listener(("127.0.0.1", args.rpc_port), authkey=authkey.encode(), backlog=n_envs)
        self.listener._listener._socket.settimeout(args.env_connect_timeout_s)  # noqa: SLF001 — fail loudly, never hang
        port = self.listener.address[1]
        gpus = (args.env_gpus or os.environ.get("CUDA_VISIBLE_DEVICES") or "0").split(",")
        py = str(pathlib.Path(args.libero_venv) / "bin/python")
        self.procs = []
        for i in range(n_envs):
            gpu = gpus[i % len(gpus)].strip()
            env = {
                **os.environ,
                "MUJOCO_GL": "egl",
                "MUJOCO_EGL_DEVICE_ID": gpu,
                "CUDA_VISIBLE_DEVICES": gpu,
                "PYTHONPATH": f"{os.environ.get('PYTHONPATH', '')}:{REPO_ROOT / 'third_party/libero'}",
            }
            for k in ("VIRTUAL_ENV", "PYTHONHOME", "XLA_PYTHON_CLIENT_MEM_FRACTION"):
                env.pop(k, None)
            cmd = [
                py,
                str(REPO_ROOT / "baselines/libero_rl_env_worker.py"),
                "--port",
                str(port),
                "--authkey",
                authkey,
                "--worker-id",
                str(i),
            ]
            self.procs.append(subprocess.Popen(cmd, env=env, cwd=str(REPO_ROOT)))
        conns: dict[int, Any] = {}
        try:
            for _ in range(n_envs):
                c = self.listener.accept()
                conns[int(msgpack_numpy.unpackb(c.recv_bytes())["hello"])] = c
        except (TimeoutError, OSError) as e:
            missing = sorted(set(range(n_envs)) - set(conns))
            dead = [i for i, p in enumerate(self.procs) if p.poll() is not None]
            for p in self.procs:
                p.kill()
            raise RuntimeError(
                f"LIBERO workers failed to connect within {args.env_connect_timeout_s}s: "
                f"missing ids {missing}, exited ids {dead}"
            ) from e
        self.conns = [conns[i] for i in range(n_envs)]
        self.pool = cf.ThreadPoolExecutor(max_workers=n_envs)
        replies = self.call_all([{"cmd": "init", "suite": suite, "language": language, "seed": 0}] * n_envs)
        self.n_init_states = int(replies[0]["n_init_states"])
        logging.info(f"[env] {n_envs} LIBERO workers up ({suite}: {language!r}, {self.n_init_states} init states)")

    def _rpc(self, i: int, msg: dict) -> dict:
        self.conns[i].send_bytes(msgpack_numpy.packb(msg))
        return msgpack_numpy.unpackb(self.conns[i].recv_bytes())

    def call(self, idx: list[int], msgs: list[dict]) -> list[dict]:
        return list(self.pool.map(self._rpc, idx, msgs))

    def call_all(self, msgs: list[dict]) -> list[dict]:
        return self.call(list(range(len(self.conns))), msgs)

    def close(self):
        for i, c in enumerate(self.conns):
            try:
                c.send_bytes(msgpack_numpy.packb({"cmd": "close"}))
                c.close()
            except OSError:
                logging.warning(f"[env] worker {i} already gone")
        for p in self.procs:
            try:
                p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill()
        self.listener.close()
        self.pool.shutdown()


class OrderedResetIds:
    """RLinf use_ordered_reset_state_ids: a seed-0 shuffled permutation of all init states, consumed
    num_group_envs at a time and reshuffled when exhausted."""

    def __init__(self, n_states: int):
        self.n = n_states
        self.rng = np.random.default_rng(seed=0)
        self._new_perm()

    def _new_perm(self):
        self.perm = np.arange(self.n)
        self.rng.shuffle(self.perm)
        self.start = 0

    def take(self, k: int) -> np.ndarray:
        if self.start + k > self.n:
            self._new_perm()
        ids = self.perm[self.start : self.start + k]
        self.start += k
        return ids


# === rollout ==================================================================================


@dataclasses.dataclass
class Sample:
    element: dict  # raw env observation (worker format), transformed lazily at train time
    chain: np.ndarray  # [K+1, H, D]
    episode: int
    n_exec: int  # executed actions from this chunk (<= replan_steps)


def collect_rollouts(
    args, gargs, env_pool, reset_ids, sample_fn, state, input_tf, output_tf, data_sharding, rng, n_chunk_steps
):
    n_envs = gargs.group_size * gargs.num_group_envs
    samples: list[Sample] = []
    ep_success: list[bool] = []
    ep_steps: list[int] = []
    n_query = 0
    for r in tqdm.tqdm(range(gargs.rollout_epoch), desc="rollout", dynamic_ncols=True):
        init_ids = np.repeat(reset_ids.take(gargs.num_group_envs), gargs.group_size)
        base_ep = len(ep_success)
        ep_success.extend([False] * n_envs)
        ep_steps.extend([0] * n_envs)
        replies = env_pool.call_all(
            [{"cmd": "reset", "init_state_id": int(s), "num_steps_wait": gargs.num_steps_wait} for s in init_ids]
        )
        obs = [rep["obs"] for rep in replies]
        active = np.ones(n_envs, dtype=bool)
        for i, rep in enumerate(replies):
            if "error" in rep:
                logging.error(f"[env {i}] reset failed:\n{rep['error']}")
                active[i] = False
        for _ in range(n_chunk_steps):
            idx = np.flatnonzero(active)
            if len(idx) == 0:
                break
            fill = list(idx) + [idx[0]] * (n_envs - len(idx))  # static batch shape for jit
            tf = [input_tf(dict(obs[i])) for i in fill]
            batch = jax.tree.map(lambda *xs: np.stack(xs), *tf)
            batch = jax.tree.map(lambda x: jax.make_array_from_process_local_data(data_sharding, x), batch)
            chains = np.asarray(sample_fn(jax.random.fold_in(rng, n_query), state, _model.Observation.from_dict(batch)))
            n_query += 1
            msgs = []
            for j, i in enumerate(idx):
                act = output_tf({"state": np.asarray(tf[j]["state"]), "actions": chains[j, -1]})["actions"]
                remaining = gargs.max_episode_steps - ep_steps[base_ep + i]
                msgs.append({"cmd": "step", "actions": np.asarray(act[: min(gargs.replan_steps, remaining)])})
            replies = env_pool.call(list(idx), msgs)
            for j, (i, rep) in enumerate(zip(idx, replies, strict=True)):
                ep = base_ep + i
                if "error" in rep:
                    logging.error(f"[env {i}] step failed (episode counted as failure):\n{rep['error']}")
                    active[i] = False
                    continue
                samples.append(Sample(element=obs[i], chain=chains[j], episode=ep, n_exec=int(rep["n_exec"])))
                obs[i] = rep["obs"]
                ep_steps[ep] += int(rep["n_exec"])
                if rep["done"]:
                    ep_success[ep] = True
                    active[i] = False
                elif ep_steps[ep] >= gargs.max_episode_steps:
                    active[i] = False
        logging.info(
            f"[rollout {r + 1}/{gargs.rollout_epoch}] success "
            f"{np.mean(ep_success[base_ep:]):.3f} init_ids={sorted(set(init_ids.tolist()))}"
        )
    return samples, np.asarray(ep_success), np.asarray(ep_steps)


def grpo_advantages(gargs: GrpoArgs, ep_success: np.ndarray):
    """Per-episode advantage + keep mask (group filter), groups = contiguous blocks of group_size."""
    scores = gargs.reward_coef * ep_success.astype(np.float64)
    groups = scores.reshape(-1, gargs.group_size)
    mean = groups.mean(axis=1, keepdims=True)
    if gargs.normalize_advantages:
        std = groups.std(axis=1, ddof=1, keepdims=True)  # torch.std default (unbiased)
        adv = (groups - mean) / (std + gargs.adv_eps)
    else:
        adv = groups
    keep = np.ones_like(mean, dtype=bool)
    if gargs.filter_rewards:
        keep = (mean >= gargs.rewards_lower_bound) & (mean <= gargs.rewards_upper_bound)
    keep = np.broadcast_to(keep, groups.shape)
    return adv.reshape(-1).astype(np.float32), keep.reshape(-1)


# === main =====================================================================================


class _AssetsShim:
    """save_state only needs data_loader.data_config() for the norm stats."""

    def __init__(self, data_config):
        self._dc = data_config

    def data_config(self):
        return self._dc


def _write_hparams(args: CliArgs, config: _config.TrainConfig, extra: dict) -> None:
    hp = {
        "exp_name": config.exp_name,
        "script": pathlib.Path(__file__).name,
        "cli_args": dataclasses.asdict(args),
        "resolved": {
            "config_name": args.config_name,
            "lora_rank": config.model.lora_rank,
            "jax_device_count": jax.device_count(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "checkpoint_dir": str(config.checkpoint_dir),
            **extra,
        },
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    p = pathlib.Path(config.checkpoint_dir) / "hparams.json"
    p.write_text(json.dumps(hp, indent=2, default=str))
    logging.info(f"Wrote hyperparameters -> {p}")


def main(args: CliArgs):
    init_logging()
    gargs = args.grpo
    logging.info(f"Running on: {platform.node()} | {args}")

    base = _config.get_config(args.config_name)
    repo_id = base.data.repo_id
    suite = suite_for_task(args.task_id)
    language = task_strings_for_ids(repo_id, [args.task_id])[0]
    pretrain_ids, holdout_ids = sample_pretrain_tasks(args.tasks_per_suite, args.split_seed)
    if args.task_id in pretrain_ids:
        logging.warning(f"task {args.task_id} is a PRETRAIN task for split seed {args.split_seed}")
    if args.task_id in args.old_task_ids:
        raise SystemExit(f"--task-id {args.task_id} is also in --old-task-ids {args.old_task_ids}")
    max_epochs = gargs.max_epochs if gargs.max_epochs is not None else (5 if suite == "libero_10" else 10)
    n_envs = gargs.group_size * gargs.num_group_envs
    n_chunk_steps = math.ceil(gargs.max_episode_steps / gargs.replan_steps)
    logging.info(
        f"[stage] task {args.task_id} ({suite}): {language!r} | old tasks {list(args.old_task_ids)} "
        f"| {max_epochs} epochs × {gargs.rollout_epoch * n_envs} episodes"
    )

    updates: dict[str, Any] = {
        "exp_name": args.exp_name,
        "overwrite": args.overwrite,
        "resume": False,
        "wandb_enabled": args.wandb_enabled,
        "seed": args.seed,
        "fsdp_devices": 1,
        "weight_loader": _weight_loaders.CheckpointWeightLoader(args.pretrained_path),
        "num_train_steps": max_epochs * gargs.updates_per_epoch,
        "keep_period": 1,
    }
    if args.checkpoint_base_dir is not None:
        updates["checkpoint_base_dir"] = args.checkpoint_base_dir
    config = dataclasses.replace(base, **updates)
    if config.model.lora_rank <= 0:
        raise SystemExit(f"config {args.config_name} has no whole-model LoRA (lora_rank=0)")
    if n_envs % jax.device_count() != 0 or gargs.micro_batch_size % jax.device_count() != 0:
        raise SystemExit(f"envs ({n_envs}) and micro_batch_size must divide #devices {jax.device_count()}")

    jax.config.update(
        "jax_compilation_cache_dir",
        str(epath.Path(os.environ.get("JAX_COMPILATION_CACHE_DIR", "~/.cache/jax")).expanduser()),
    )
    rng = jax.random.key(config.seed)
    rollout_rng, init_rng = jax.random.split(rng)
    np_rng = np.random.default_rng(config.seed)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    mngr, _ = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir, keep_period=config.keep_period, overwrite=config.overwrite, resume=False
    )
    _write_hparams(
        args,
        config,
        {
            "task_id": args.task_id,
            "task_language": language,
            "suite": suite,
            "max_epochs": max_epochs,
            "old_task_ids": list(args.old_task_ids),
            "n_envs": n_envs,
            "n_chunk_steps": n_chunk_steps,
            "pretrained_path": args.pretrained_path,
        },
    )
    if config.wandb_enabled:
        wandb.init(
            name=config.exp_name,
            project=config.project_name,
            config={"cli": dataclasses.asdict(args), "task_language": language, "max_epochs": max_epochs},
        )
    else:
        wandb.init(mode="disabled")

    # Same inference transform stack as policy_config.create_trained_policy (what serve_policy/eval uses).
    data_config = config.data.create(config.assets_dirs, config.model)
    input_tf = _transforms.compose(
        [
            _transforms.InjectDefaultPrompt(None),
            *data_config.data_transforms.inputs,
            _transforms.Normalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ]
    )
    output_tf = _transforms.compose(
        [
            *data_config.model_transforms.outputs,
            _transforms.Unnormalize(data_config.norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ]
    )

    tx = optax.chain(
        optax.clip_by_global_norm(gargs.clip_grad),
        optax.adamw(gargs.lr, b1=gargs.adam_b1, b2=gargs.adam_b2, eps=gargs.adam_eps, weight_decay=gargs.weight_decay),
    )
    state, state_sharding = init_train_state(config, init_rng, mesh, tx)
    jax.block_until_ready(state)
    n_lora = sum(
        int(np.prod(v.shape)) for v in jax.tree.leaves(state.params.filter(config.trainable_filter).to_pure_dict())
    )
    logging.info(f"Initialized train state; trainable (LoRA) params: {n_lora / 1e6:.1f}M")

    batch_keys_sharding = dict.fromkeys(("obs", "chain", "old_logp", "adv", "act_mask", "weight"), data_sharding)
    sample_fn = jax.jit(
        functools.partial(sample_chain, gargs),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=replicated,
    )
    old_logp_fn = jax.jit(
        functools.partial(old_logprob_step, gargs),
        in_shardings=(state_sharding, data_sharding),
        out_shardings=replicated,
    )
    grad_fn = jax.jit(
        functools.partial(grad_step, gargs, config),
        in_shardings=(state_sharding, {**batch_keys_sharding, "normalizer": replicated}),
        out_shardings=(replicated, replicated),
    )
    add_fn = jax.jit(lambda a, b: jax.tree.map(jnp.add, a, b))
    apply_fn = jax.jit(
        functools.partial(apply_step, config),
        in_shardings=(state_sharding, replicated),
        out_shardings=(state_sharding, replicated),
        donate_argnums=(0,),
    )

    env_pool = EnvPool(args, n_envs, suite, language)
    reset_ids = OrderedResetIds(env_pool.n_init_states)
    metrics_file = open(pathlib.Path(config.checkpoint_dir) / "train_metrics.jsonl", "a")  # noqa: SIM115
    K, R = gargs.num_denoise_steps, gargs.replan_steps
    tokens_per_sample = K * R * LIBERO_ACTION_DIM
    mb = gargs.micro_batch_size

    def to_batch(rows: list[Sample], extra: dict[str, np.ndarray]) -> dict:
        obs = jax.tree.map(lambda *xs: np.stack(xs), *[input_tf(dict(s.element)) for s in rows])
        obs.pop("actions", None)
        b = {"obs": obs, "chain": np.stack([s.chain for s in rows]), **extra}
        return jax.tree.map(lambda x: jax.make_array_from_process_local_data(data_sharding, x), b)

    try:
        for epoch in range(1, max_epochs + 1):
            t0 = time.time()
            samples, ep_success, ep_steps = collect_rollouts(
                args,
                gargs,
                env_pool,
                reset_ids,
                sample_fn,
                state,
                input_tf,
                output_tf,
                data_sharding,
                jax.random.fold_in(rollout_rng, epoch),
                n_chunk_steps,
            )
            t_roll = time.time() - t0
            adv_ep, keep_ep = grpo_advantages(gargs, ep_success)
            ratio_ep = np.maximum(ep_steps, 1) / gargs.max_episode_steps  # loss_mask_sum / max_episode_steps
            train_idx = np.array([i for i, s in enumerate(samples) if keep_ep[s.episode]], dtype=np.int64)
            n_eps = len(ep_success)
            padded_total = n_eps * n_chunk_steps  # RLinf samples incl. post-done padding
            normalizer = np.float32(padded_total / gargs.updates_per_epoch * tokens_per_sample)
            info = {
                "epoch": epoch,
                "rollout/success_rate": float(ep_success.mean()),
                "rollout/episode_len": float(ep_steps.mean()),
                "rollout/n_samples": len(samples),
                "rollout/kept_group_frac": float(keep_ep.reshape(-1, gargs.group_size)[:, 0].mean()),
                "rollout/n_train_samples": len(train_idx),
                "time/rollout_s": t_roll,
            }
            logging.info(f"[epoch {epoch}] rollout done: {info}")

            if len(train_idx) > 0:
                t1 = time.time()
                np_rng.shuffle(train_idx)
                n_pad = (-len(train_idx)) % mb
                order = np.concatenate([train_idx, np.full(n_pad, train_idx[0])])
                live = np.concatenate([np.ones(len(train_idx), bool), np.zeros(n_pad, bool)])
                # old log-probs under the rollout params (computed once, before any update)
                old_logp = np.empty((len(order), K, R, LIBERO_ACTION_DIM), np.float32)
                for s0 in tqdm.tqdm(range(0, len(order), mb), desc="old_logp", dynamic_ncols=True):
                    rows = [samples[i] for i in order[s0 : s0 + mb]]
                    old_logp[s0 : s0 + mb] = np.asarray(old_logp_fn(state, to_batch(rows, {})))

                def extra_for(
                    sl, _order=order, _live=live, _old=old_logp, _samples=samples, _adv=adv_ep, _ratio=ratio_ep
                ):
                    idx = _order[sl]
                    eps = np.array([_samples[i].episode for i in idx])
                    n_exec = np.array([_samples[i].n_exec for i in idx])
                    return {
                        "old_logp": _old[sl],
                        "adv": _adv[eps],
                        "act_mask": (np.arange(R)[None, :] < n_exec[:, None]).astype(np.float32),
                        "weight": np.where(_live[sl], 1.0 / _ratio[eps], 0.0).astype(np.float32),
                    }

                n_mb = len(order) // mb
                per_update = np.array_split(np.arange(n_mb), min(gargs.updates_per_epoch, n_mb))
                acc = {"loss": 0.0, "n_tokens": 0.0, "clipfrac_sum": 0.0, "approx_kl_sum": 0.0, "ratio_sum": 0.0}
                gnorms = []
                for u, mbs in enumerate(per_update):
                    grads = None
                    for m in tqdm.tqdm(mbs, desc=f"update {u + 1}/{len(per_update)}", dynamic_ncols=True):
                        sl = slice(m * mb, (m + 1) * mb)
                        batch = to_batch([samples[i] for i in order[sl]], extra_for(sl))
                        batch["normalizer"] = jnp.asarray(normalizer)
                        g, aux = grad_fn(state, batch)
                        grads = g if grads is None else add_fn(grads, g)
                        for k2 in acc:
                            acc[k2] += float(aux[k2])
                    state, gnorm = apply_fn(state, grads)
                    gnorms.append(float(gnorm))
                ntok = max(acc["n_tokens"], 1.0)
                info.update(
                    {
                        "train/loss": acc["loss"] / len(per_update),
                        "train/clip_frac": acc["clipfrac_sum"] / ntok,
                        "train/approx_kl": acc["approx_kl_sum"] / ntok,
                        "train/ratio": acc["ratio_sum"] / ntok,
                        "train/grad_norm": float(np.mean(gnorms)),
                        "train/n_updates": len(per_update),
                        "time/train_s": time.time() - t1,
                    }
                )
            else:
                logging.warning(f"[epoch {epoch}] every group filtered (all-fail or all-success): no update")

            logging.info(
                f"[epoch {epoch}] "
                + ", ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in info.items())
            )
            metrics_file.write(json.dumps(info) + "\n")
            metrics_file.flush()
            wandb.log(info, step=epoch)
            _checkpoints.save_state(mngr, state, _AssetsShim(data_config), epoch)
            samples = None  # free the rollout buffer before the next epoch's collection
    finally:
        env_pool.close()
        metrics_file.close()
    logging.info("Waiting for checkpoint manager to finish")
    mngr.wait_until_finished()


if __name__ == "__main__":
    main(tyro.cli(CliArgs))

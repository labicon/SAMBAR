r"""One LIBERO simulator for online RL (runs in examples/libero/.venv, NOT the openpi venv).

Spawned by -m baselines.train_grpo — one process per parallel env. It connects
back to the trainer over a stdlib `multiprocessing.connection` socket and serves a tiny RPC
(msgpack-numpy frames, so numpy arrays cross the two venvs without pickle-version coupling):

    {"cmd": "init",  "suite": str, "language": str, "seed": int}      -> {"n_init_states": int}
    {"cmd": "reset", "init_state_id": int, "num_steps_wait": int}      -> {"obs": element}
    {"cmd": "step",  "actions": float[n, 7]}                           -> {"obs", "done", "n_exec"}
    {"cmd": "close"}

`element` is built EXACTLY like examples/libero/main.py (180°-rotated agentview + wrist image,
resize_with_pad to 224, uint8; 8-D eef pos/axis-angle/gripper state; prompt = task.language), so
the RL policy sees the same inputs as the openpi LIBERO eval. `done` is LIBERO's success signal;
`step` stops executing the chunk at the first success (env termination, as in the paper's
auto_reset=False / ignore_terminations=False setting).
"""

from __future__ import annotations

import argparse
import math
from multiprocessing.connection import Client
import pathlib
import traceback

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import msgpack_numpy

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
RESIZE_SIZE = 224


def _quat2axisangle(quat):
    """Copied from examples/libero/main.py (robosuite transform_utils)."""
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _element(obs, prompt: str) -> dict:
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, RESIZE_SIZE, RESIZE_SIZE))
    wrist_img = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist_img, RESIZE_SIZE, RESIZE_SIZE))
    return {
        "observation/image": img,
        "observation/wrist_image": wrist_img,
        "observation/state": np.concatenate(
            (obs["robot0_eef_pos"], _quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ).astype(np.float32),
        "prompt": prompt,
    }


class Worker:
    def __init__(self):
        self.env = None
        self.init_states = None
        self.prompt = None
        self.last_obs = None

    def init(self, suite: str, language: str, seed: int) -> dict:
        task_suite = benchmark.get_benchmark_dict()[suite]()
        matches = [i for i in range(task_suite.n_tasks) if task_suite.get_task(i).language == language]
        if len(matches) != 1:
            raise ValueError(f"expected exactly one task in {suite} with language {language!r}, got {matches}")
        task = task_suite.get_task(matches[0])
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        if self.env is not None:
            self.env.close()
        self.env = OffScreenRenderEnv(
            bddl_file_name=bddl, camera_heights=LIBERO_ENV_RESOLUTION, camera_widths=LIBERO_ENV_RESOLUTION
        )
        self.env.seed(seed)
        self.init_states = task_suite.get_task_init_states(matches[0])
        self.prompt = str(task.language)
        return {"n_init_states": len(self.init_states)}

    def reset(self, init_state_id: int, num_steps_wait: int) -> dict:
        self.env.reset()
        obs = self.env.set_init_state(self.init_states[init_state_id])
        for _ in range(num_steps_wait):
            obs, _, _, _ = self.env.step(LIBERO_DUMMY_ACTION)
        self.last_obs = obs
        return {"obs": _element(obs, self.prompt)}

    def step(self, actions: np.ndarray) -> dict:
        done = False
        n_exec = 0
        obs = self.last_obs
        for a in np.asarray(actions):
            obs, _, done, _ = self.env.step(a.tolist())
            n_exec += 1
            if done:
                break
        self.last_obs = obs
        return {"obs": _element(obs, self.prompt), "done": bool(done), "n_exec": n_exec}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--authkey", required=True)
    parser.add_argument("--worker-id", type=int, required=True)
    args = parser.parse_args()

    conn = Client((args.host, args.port), authkey=args.authkey.encode())
    conn.send_bytes(msgpack_numpy.packb({"hello": args.worker_id}))
    worker = Worker()
    while True:
        msg = msgpack_numpy.unpackb(conn.recv_bytes())
        cmd = msg["cmd"]
        if cmd == "close":
            if worker.env is not None:
                worker.env.close()
            break
        try:
            if cmd == "init":
                reply = worker.init(msg["suite"], msg["language"], int(msg["seed"]))
            elif cmd == "reset":
                reply = worker.reset(int(msg["init_state_id"]), int(msg["num_steps_wait"]))
            elif cmd == "step":
                reply = worker.step(msg["actions"])
            else:
                raise ValueError(f"unknown cmd {cmd}")
        except Exception:  # surface sim errors to the trainer (treated as episode failure)
            reply = {"error": traceback.format_exc()}
        conn.send_bytes(msgpack_numpy.packb(reply))
    conn.close()


if __name__ == "__main__":
    main()

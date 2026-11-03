"""Serve a LIBERO policy checkpoint for simulator evaluation."""

import dataclasses
import logging

import tyro

from openpi.policies import policy_config
from openpi.serving.websocket_policy_server import WebsocketPolicyServer
from openpi.training.config import get_config


@dataclasses.dataclass
class Args:
    checkpoint: str
    config: str = "sambar_narrow"
    port: int = 8000
    host: str = "0.0.0.0"


def main(args):
    policy = policy_config.create_trained_policy(get_config(args.config), args.checkpoint)
    WebsocketPolicyServer(policy, host=args.host, port=args.port, metadata=policy.metadata).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))

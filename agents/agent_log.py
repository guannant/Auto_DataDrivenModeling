import json
import time
from pathlib import Path

import numpy as np


class AgentLogger:
    """Append one JSON record per line to agent_log.jsonl."""

    def __init__(self, path):
        self.path = Path(path)

    def log(self, **record):
        record = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **record}
        with open(self.path, "a") as f:
            f.write(json.dumps(record, default=_to_json, ensure_ascii=False) + "\n")


def _to_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return str(obj)


def log_agent(state, **record):
    """
    Write one record if the state has an agent logger.
    `generation` is the number of completed generations when the agent ran.
    """
    logger = state.get("agent_logger")
    if logger is not None:
        logger.log(generation=state.get("generation"), **record)

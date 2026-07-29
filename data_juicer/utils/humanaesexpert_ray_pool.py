"""Detached Ray actors that keep HumanAesExpert models warm across jobs."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List


def actor_name(prefix: str, index: int) -> str:
    return f"{prefix}-{index:02d}"


def config_fingerprint(config: Dict[str, Any]) -> str:
    encoded = json.dumps(
        config,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class HumanAesExpertScoringActor:
    """Own one model replica and score requests serially on its assigned GPU."""

    def __init__(self, scorer_config: Dict[str, Any]):
        from data_juicer.ops.mapper.image_humanaesexpert_mapper import (
            ImageHumanAesExpertMapper,
        )

        self._config = dict(scorer_config)
        self._fingerprint = config_fingerprint(self._config)
        self._scorer = ImageHumanAesExpertMapper(
            **self._config,
            num_proc=1,
            num_gpus=1,
            delete_local_cache_after_processing=False,
            persistent_actor_pool=False,
        )

    def warmup(self) -> Dict[str, Any]:
        import ray

        self._scorer._load_model()
        return {
            "config_fingerprint": self._fingerprint,
            "gpu_ids": ray.get_gpu_ids(),
            "model": self._scorer.model_name_or_path,
        }

    def identity(self) -> Dict[str, Any]:
        return {
            "config_fingerprint": self._fingerprint,
            "model": self._scorer.model_name_or_path,
        }

    def score_image(self, path: str) -> Dict[str, Any]:
        return self._scorer._score_image(path)


def get_pool_handles(
    ray_module,
    namespace: str,
    prefix: str,
    size: int,
) -> List[Any]:
    handles = []
    for index in range(size):
        handles.append(
            ray_module.get_actor(
                actor_name(prefix, index),
                namespace=namespace,
            )
        )
    return handles


def start_pool(
    ray_module,
    namespace: str,
    prefix: str,
    size: int,
    scorer_config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Create missing detached actors and warm all model replicas."""
    expected_fingerprint = config_fingerprint(scorer_config)
    remote_actor = ray_module.remote(
        num_gpus=1,
        max_restarts=0,
        max_task_retries=0,
    )(HumanAesExpertScoringActor)
    handles = []
    for index in range(size):
        name = actor_name(prefix, index)
        try:
            handle = ray_module.get_actor(name, namespace=namespace)
        except ValueError:
            handle = remote_actor.options(
                name=name,
                namespace=namespace,
                lifetime="detached",
            ).remote(scorer_config)
        handles.append(handle)

    identities = ray_module.get(
        [handle.identity.remote() for handle in handles]
    )
    mismatched = [
        actor_name(prefix, index)
        for index, identity in enumerate(identities)
        if identity.get("config_fingerprint") != expected_fingerprint
    ]
    if mismatched:
        raise RuntimeError(
            "Existing HumanAesExpert actors use a different model "
            f"configuration: {mismatched}. Stop the pool before restarting."
        )
    return ray_module.get([handle.warmup.remote() for handle in handles])


def stop_pool(
    ray_module,
    namespace: str,
    prefix: str,
    size: int,
) -> int:
    stopped = 0
    for index in range(size):
        try:
            handle = ray_module.get_actor(
                actor_name(prefix, index),
                namespace=namespace,
            )
        except ValueError:
            continue
        ray_module.kill(handle, no_restart=True)
        stopped += 1
    return stopped

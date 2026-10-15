"""vLLM plugin: tiered sleep/wake for multi-LLM agent restore.

Loaded in every vLLM process through the ``vllm.general_plugins`` entry point.
It registers the ``tiered`` sleep-mode backend (backend.py). When TIERED_WEIGHTS
is set, it also does three things:

1. Selects ``tiered`` in place of the default ``cumem`` backend. vLLM 0.30
   has no CLI flag for ``sleep_mode_backend``.
2. Keeps the KV of paused requests (unless TIERED_KV=recompute). ``sleep(level=1, mode="keep")`` normally
   resets the prefix cache with ``reset_running_requests=True``, which preempts
   the frozen requests and throws their KV away. Under PAUSED_ALL ("keep") the
   cache reset is now skipped, so the requests resume on the restored KV.
3. Records the virtual KV tensors that kvcached creates, so the backend can
   find the bytes behind each in-use kvcached page.

Configuration (environment):
  TIERED_WEIGHTS   dram | ssd | gds   where weights go on sleep
  TIERED_KV        dram | ssd | gds   where in-use KV pages go on sleep;
                   recompute: KV is not kept (RECOMPUTE preemption): vLLM's
                   cache reset preempts the frozen requests, which re-prefill on wake
  TIERED_PARALLEL  1 (default) loads weights and KV concurrently on wake; 0 loads
                   weights first, then KV
  TIERED_DIR       directory for ssd/gds files (one per instance)
  TIERED_LOG       JSON-lines file, one line per suspend/resume
  TIERED_IO_THREADS, TIERED_CHUNK_MB   I/O parallelism and chunk size (8, 64)
  TIERED_GDS_CHUNK_MB   GDS chunk size, registered per I/O (256)
  TIERED_DRAM_KV_RESERVE_GB   pinned DRAM reserved for KV at the first sleep (24)
"""

import os

_PATCHED = "_tiered_restore_patched"


def register() -> None:
    from vllm.device_allocator.sleep_mode_backend import SleepModeBackendFactory

    if "tiered" not in SleepModeBackendFactory._registry:
        SleepModeBackendFactory.register_backend("tiered", "tiered_restore.backend", "TieredBackend")
    if not os.environ.get("TIERED_WEIGHTS") or getattr(SleepModeBackendFactory, _PATCHED, False):
        return
    setattr(SleepModeBackendFactory, _PATCHED, True)
    _select_tiered_backend(SleepModeBackendFactory)
    if os.environ.get("TIERED_KV") != "recompute":
        _keep_kv_on_keep_pause()
    _record_kvcached_tensors()


def _select_tiered_backend(factory) -> None:
    from vllm.logger import init_logger

    logger = init_logger("vllm.tiered_restore")

    def create_backend(cls, model_config):
        logger.info("Using sleep-mode backend: tiered (weights=%s, kv=%s, parallel=%s)",
                    os.environ["TIERED_WEIGHTS"], os.environ.get("TIERED_KV", "dram"),
                    os.environ.get("TIERED_PARALLEL", "1"))
        return cls.get_backend_class("tiered")()

    factory.create_backend = classmethod(create_backend)


def _keep_kv_on_keep_pause() -> None:
    from vllm.v1.core.sched.interface import PauseState
    from vllm.v1.engine.core import EngineCore

    finish_pause = EngineCore._finish_pause

    def _finish_pause(self, clear_cache: bool) -> None:
        if self.scheduler.pause_state == PauseState.PAUSED_ALL:
            clear_cache = False
        finish_pause(self, clear_cache)

    EngineCore._finish_pause = _finish_pause


# group_id -> tensors returned by kvcached's create_kv_tensors
KV_TENSORS: dict[int, list] = {}


def _record_kvcached_tensors() -> None:
    try:
        from kvcached.integration.vllm import interfaces
    except ImportError:
        return
    create = interfaces.create_kv_tensors

    def create_kv_tensors(*args, group_id: int = 0, **kwargs):
        tensors = create(*args, group_id=group_id, **kwargs)
        KV_TENSORS[group_id] = list(tensors)
        return tensors

    interfaces.create_kv_tensors = create_kv_tensors

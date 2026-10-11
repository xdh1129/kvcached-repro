"""On-demand weight swapping through host DRAM, in front of vLLM sleep-mode instances.

Time-sharing baseline in the style of the Prism paper: every instance starts asleep
(launch_servers.sh with SLEEP_AFTER_START=1). A request for an asleep instance
wakes it (weights CPU DRAM -> GPU, vLLM /wake_up). If the GPU lacks room, the
least-recently-used awake instance on that GPU with no request in flight is put
to sleep first (vLLM /sleep?level=1: weights GPU -> CPU DRAM, KV discarded).
If every awake instance is busy, one is drained: the awake instance with the
fewest requests in flight (then least recently used) stops admitting requests,
finishes its in-flight ones, and is put to sleep. Its new requests queue until
it is woken again. Draining bounds the wait; without it a continuously busy
neighbour would starve the instance waiting to wake.

Room check: NVML free memory on the GPU >= the instance's wake size + headroom.
Wake size per instance = memory awake - memory asleep, from startup.tsv.

Requests: POST /<instance name>/v1/completions (body passed through unchanged).
Writes to <run_dir>:
  swaps.jsonl           one line per sleep/wake: t (since proxy start), t_unix, gpu,
                        instance, action, reason (instance the swap made room
                        for), duration_s, drained_s (sleep after a drain: time
                        from drain start to sleep), gpu_used_mib_before/after
  proxy_requests.jsonl  one line per request: id (X-Request-Id), instance,
                        arrive_s, ready_s, swap_wait_s, cold (needed a wake),
                        evictions (sleeps done to make room), done_s, status

usage: python swap_proxy.py <run_dir> [--port 30999] [--headroom-gib 3]
"""

import argparse
import asyncio
import csv
import json
import os
import time

import aiohttp
import pynvml
from aiohttp import web

MIB = 1024 * 1024


class Instance:
    def __init__(self, name, gpu, port, wake_mib):
        self.name, self.gpu, self.port, self.wake_mib = name, gpu, port, wake_mib
        self.awake = False
        self.draining_since = None  # set while being drained for eviction
        self.inflight = 0
        self.last_used = 0.0

    @property
    def serving(self):
        return self.awake and self.draining_since is None


class Proxy:
    def __init__(self, run_dir, headroom_mib):
        self.run_dir = run_dir
        self.headroom_mib = headroom_mib
        self.t0 = time.time()
        self.instances = {}
        with open(os.path.join(run_dir, "startup.tsv")) as f:
            for row in csv.DictReader(f, delimiter="\t"):
                wake = int(row["instance_mib"]) - int(row["instance_asleep_mib"])
                self.instances[row["name"]] = Instance(row["name"], int(row["gpu"]), int(row["port"]), wake)
        pynvml.nvmlInit()
        self.handles = {g: pynvml.nvmlDeviceGetHandleByIndex(g) for g in {i.gpu for i in self.instances.values()}}
        self.cond = {g: asyncio.Condition() for g in self.handles}
        self.swaps = open(os.path.join(run_dir, "swaps.jsonl"), "a")
        self.reqlog = open(os.path.join(run_dir, "proxy_requests.jsonl"), "a")
        self.session = None

    def used_free_mib(self, gpu):
        mem = pynvml.nvmlDeviceGetMemoryInfo(self.handles[gpu])
        return mem.used // MIB, mem.free // MIB

    async def _swap(self, inst, action, reason):
        before, _ = self.used_free_mib(inst.gpu)
        if action == "sleep":
            inst.awake = False  # before the await: new requests must not reach a sleeping engine
        t = time.time()
        url = f"http://localhost:{inst.port}/" + ("wake_up" if action == "wake" else "sleep?level=1")
        async with self.session.post(url) as r:
            if r.status != 200:
                raise RuntimeError(f"{action} {inst.name}: HTTP {r.status} {await r.text()}")
        dur = time.time() - t
        inst.awake = action == "wake"
        drained = round(t - inst.draining_since, 4) if inst.draining_since else None
        inst.draining_since = None
        after, _ = self.used_free_mib(inst.gpu)
        self.swaps.write(json.dumps({
            "t": round(t - self.t0, 4), "t_unix": round(t, 3), "gpu": inst.gpu, "instance": inst.name,
            "action": action, "reason": reason, "duration_s": round(dur, 4), "drained_s": drained,
            "gpu_used_mib_before": before, "gpu_used_mib_after": after}) + "\n")
        self.swaps.flush()

    async def ensure_awake(self, inst):
        """Make inst serving: wake it, evicting (or draining) neighbours for room.

        Returns (woke, evictions): whether this call did the wake, and how many
        instances it put to sleep. Callers that queued behind another caller's
        wake get (False, 0) but still record their wait.
        """
        if inst.serving:
            return False, 0
        woke, evictions = False, 0
        cond = self.cond[inst.gpu]
        async with cond:
            while not inst.serving:
                if inst.awake:  # being drained for eviction: wait until it sleeps, then wake it
                    await cond.wait()
                    continue
                _, free = self.used_free_mib(inst.gpu)
                if free >= inst.wake_mib + self.headroom_mib:
                    await self._swap(inst, "wake", inst.name)
                    woke = True
                    cond.notify_all()
                    break
                awake = [i for i in self.instances.values() if i.gpu == inst.gpu and i.awake and i is not inst]
                idle = [i for i in awake if i.inflight == 0]
                if idle:
                    victim = min(idle, key=lambda i: (i.draining_since is None, i.last_used))
                    await self._swap(victim, "sleep", inst.name)
                    evictions += 1
                    cond.notify_all()
                    continue
                if awake and not any(i.draining_since for i in awake):
                    victim = min(awake, key=lambda i: (i.inflight, i.last_used))
                    victim.draining_since = time.time()
                await cond.wait()  # an in-flight request on this GPU finishes
        return woke, evictions

    async def handle(self, request):
        inst = self.instances.get(request.match_info["name"])
        if inst is None:
            return web.Response(status=404, text="unknown instance")
        body = await request.read()
        rec = {"id": request.headers.get("X-Request-Id"), "instance": inst.name,
               "arrive_s": round(time.time() - self.t0, 4)}
        try:
            rec["cold"], rec["evictions"] = await self.ensure_awake(inst)
            # No await between the awake check above and this increment, so a
            # concurrent eviction cannot pick this instance in between.
            inst.inflight += 1
        except Exception as e:
            rec.update(status=f"swap error: {e}")
            self._log(rec)
            return web.Response(status=503, text=str(e))
        rec["ready_s"] = round(time.time() - self.t0, 4)
        rec["swap_wait_s"] = round(rec["ready_s"] - rec["arrive_s"], 4)
        resp = web.StreamResponse()
        try:
            async with self.session.post(f"http://localhost:{inst.port}/{request.match_info['path']}",
                                         data=body, headers={"Content-Type": "application/json"}) as up:
                resp.set_status(up.status)
                resp.content_type = up.content_type
                await resp.prepare(request)
                async for chunk in up.content.iter_any():
                    await resp.write(chunk)
                rec["status"] = up.status
            await resp.write_eof()
        except Exception as e:
            rec["status"] = f"proxy error: {type(e).__name__}: {e}"
        finally:
            inst.inflight -= 1
            inst.last_used = time.time()
            rec["done_s"] = round(time.time() - self.t0, 4)
            self._log(rec)
            async with self.cond[inst.gpu]:
                self.cond[inst.gpu].notify_all()
        return resp

    def _log(self, rec):
        self.reqlog.write(json.dumps(rec) + "\n")
        self.reqlog.flush()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--port", type=int, default=30999)
    ap.add_argument("--headroom-gib", type=float, default=3.0,
                    help="free memory kept beyond the wake size, for KV growth and activations")
    args = ap.parse_args()

    proxy = Proxy(args.run_dir, int(args.headroom_gib * 1024))
    proxy.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600),
                                          connector=aiohttp.TCPConnector(limit=0))
    app = web.Application(client_max_size=64 * MIB)
    app.router.add_get("/health", lambda _: web.Response(text="ok"))
    app.router.add_post("/{name}/{path:.*}", proxy.handle)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "localhost", args.port).start()
    print(f"swap proxy on :{args.port}; wake sizes (MiB): "
          + ", ".join(f"{i.name}={i.wake_mib}" for i in proxy.instances.values()), flush=True)
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())

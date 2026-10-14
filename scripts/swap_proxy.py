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

--keep (agent restore, with the tiered_restore plugin): eviction does not wait
for in-flight requests. The victim sleeps with /sleep?level=1&mode=keep: its
requests freeze, and its weights and their KV are offloaded. Waking restores
both, and the requests continue. A woken instance stays resident at least
--min-resident-s while it has requests in flight, so two busy instances do not
evict each other on every swap. Idle instances can be evicted at any time. The
victim is the least recently used candidate (last request arrival or finish).
With the plugin's TIERED_KV=recompute, the frozen requests' KV is dropped
instead and they re-prefill on wake (Prism's RECOMPUTE preemption).
An instance put to sleep with requests in flight queues its own wake.
Its wake size is then what the sleep released (weights + KV), measured per
process group with NVML.

--kv-pressure (with --keep): time-shares a GPU whose busy instances are short of
KV memory. Without it, eviction happens only when an asleep instance has to wake;
instances that stay awake and compete for KV are left to vLLM, which preempts and
recomputes inside each instance. Every --poll-s, per GPU:
  starved   an awake instance with queued requests (vllm:num_requests_waiting > 0)
            for >= --starve-s, or an instance asleep with frozen requests
  pressure  NVML free memory <= --pressure-free-gib, and >= 2 instances with demand
            (busy awake + frozen asleep)
  action    sleep (mode=keep) the least recently used busy awake instance among
            those resident >= --min-resident-s, unless it is the only starved one. At most one such sleep per --pressure-cooldown-s.
An instance frozen this way stays asleep at least --min-resident-s (one time
slice for the others), then wakes only by evicting idle instances or once memory
frees up. Busy instances are evicted only by this rule, so two instances cannot
evict each other back and forth.

Requests: POST /<instance name>/v1/completions (body passed through unchanged).
Writes to <run_dir>:
  swaps.jsonl           one line per sleep/wake: t (since proxy start), t_unix, gpu,
                        instance, action, reason (instance the swap made room
                        for), duration_s, drained_s (sleep after a drain: time
                        from drain start to sleep), gpu_used_mib_before/after,
                        inflight (requests frozen by a keep sleep / resumed by
                        the wake), need_mib (wake size used for the next wake);
                        kv-pressure sleeps have reason "kv_pressure:<starved>=<queued>,..."
  proxy_requests.jsonl  one line per request: id (X-Request-Id), instance,
                        arrive_s, ready_s, swap_wait_s, cold (needed a wake),
                        evictions (sleeps done to make room), done_s, status

--schedule <ref run_dir> (with --keep): replays the sleeps and wakes of a reference
run instead of deciding them, so that configs that differ only in the offload
path see the same evictions at the same times. Times are taken relative to the
first request arrival of each run. Per GPU, actions run in order, each at its
scheduled time or as soon as the previous one finishes (if a slower path falls
behind). A wake also waits until the GPU has room for it. A request for an
asleep instance waits for that instance's next scheduled wake; if none is left,
the normal policy wakes it. A scheduled sleep of an instance that was idle in
the reference first waits for this run's in-flight requests on it to finish.
The LRU and KV-pressure policies are off while replaying. Each replayed swap records scheduled_s and lag_s in swaps.jsonl.

usage: python swap_proxy.py <run_dir> [--port 30999] [--headroom-gib 3]
                            [--keep [--min-resident-s 5] [--kv-pressure [--pressure-free-gib 6]]
                                    [--schedule REF_RUN_DIR]]
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
    def __init__(self, name, gpu, port, wake_mib, pgid):
        self.name, self.gpu, self.port, self.wake_mib, self.pgid = name, gpu, port, wake_mib, pgid
        self.need_mib = wake_mib  # memory the next wake brings back (weights, plus frozen KV under --keep)
        self.resident_since = 0.0
        self.asleep_since = 0.0
        self.awake = False
        self.draining_since = None  # set while being drained for eviction
        self.inflight = 0
        self.last_used = 0.0

    @property
    def serving(self):
        return self.awake and self.draining_since is None


class Proxy:
    def __init__(self, run_dir, headroom_mib, keep=False, min_resident_s=5.0, kv_pressure=False,
                 pressure_free_mib=6144, starve_s=1.0, pressure_cooldown_s=2.0, poll_s=0.5):
        self.run_dir = run_dir
        self.headroom_mib = headroom_mib
        self.keep, self.min_resident_s = keep, min_resident_s
        self.schedule = None  # set by load_schedule (--schedule)
        self.kv_pressure, self.pressure_free_mib = kv_pressure, pressure_free_mib
        self.starve_s, self.pressure_cooldown_s, self.poll_s = starve_s, pressure_cooldown_s, poll_s
        self.t0 = time.time()
        self.instances = {}
        with open(os.path.join(run_dir, "servers.tsv")) as f:
            pgids = {name: int(pgid) for name, _, pgid in csv.reader(f, delimiter="\t")}
        with open(os.path.join(run_dir, "startup.tsv")) as f:
            for row in csv.DictReader(f, delimiter="\t"):
                wake = int(row["instance_mib"]) - int(row["instance_asleep_mib"])
                self.instances[row["name"]] = Instance(row["name"], int(row["gpu"]), int(row["port"]), wake,
                                                       pgids[row["name"]])
        pynvml.nvmlInit()
        self.handles = {g: pynvml.nvmlDeviceGetHandleByIndex(g) for g in {i.gpu for i in self.instances.values()}}
        self.cond = {g: asyncio.Condition() for g in self.handles}
        self.swaps = open(os.path.join(run_dir, "swaps.jsonl"), "a")
        self.reqlog = open(os.path.join(run_dir, "proxy_requests.jsonl"), "a")
        self.session = None

    def used_free_mib(self, gpu):
        mem = pynvml.nvmlDeviceGetMemoryInfo(self.handles[gpu])
        return mem.used // MIB, mem.free // MIB

    def instance_mib(self, inst):
        """GPU memory held by the instance's process group."""
        total = 0
        for p in pynvml.nvmlDeviceGetComputeRunningProcesses(self.handles[inst.gpu]):
            try:
                if os.getpgid(p.pid) == inst.pgid:
                    total += (p.usedGpuMemory or 0) // MIB
            except ProcessLookupError:
                pass
        return total

    async def _swap(self, inst, action, reason, extra=None):
        before, _ = self.used_free_mib(inst.gpu)
        inst_before = self.instance_mib(inst)
        inflight = inst.inflight
        if action == "sleep":
            inst.awake = False  # before the await: new requests must not reach a sleeping engine
        t = time.time()
        sleep = "sleep?level=1" + ("&mode=keep" if self.keep else "")
        url = f"http://localhost:{inst.port}/" + ("wake_up" if action == "wake" else sleep)
        async with self.session.post(url) as r:
            if r.status != 200:
                raise RuntimeError(f"{action} {inst.name}: HTTP {r.status} {await r.text()}")
        dur = time.time() - t
        inst.awake = action == "wake"
        if action == "wake":
            inst.resident_since = time.time()
        else:
            inst.asleep_since = time.time()
        if action == "sleep" and self.keep:
            inst.need_mib = max(inst.wake_mib, inst_before - self.instance_mib(inst))
        drained = round(t - inst.draining_since, 4) if inst.draining_since else None
        inst.draining_since = None
        after, _ = self.used_free_mib(inst.gpu)
        self.swaps.write(json.dumps({
            "t": round(t - self.t0, 4), "t_unix": round(t, 3), "gpu": inst.gpu, "instance": inst.name,
            "action": action, "reason": reason, "duration_s": round(dur, 4), "drained_s": drained,
            "gpu_used_mib_before": before, "gpu_used_mib_after": after,
            "inflight": inflight, "need_mib": inst.need_mib, **(extra or {})}) + "\n")
        self.swaps.flush()

    async def ensure_awake(self, inst, evict_busy=True):
        """Make inst serving: wake it, evicting (or draining) neighbours for room.

        Returns (woke, evictions): whether this call did the wake, and how many
        instances it put to sleep. Callers that queued behind another caller's
        wake get (False, 0) but still record their wait.
        evict_busy=False (--keep only): make room by evicting idle instances only.
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
                if not evict_busy and time.time() - inst.asleep_since < self.min_resident_s:
                    # a frozen instance sits out one time slice, so the instances it made room for can use it
                    try:
                        await asyncio.wait_for(cond.wait(), timeout=0.5)
                    except TimeoutError:
                        pass
                    continue
                _, free = self.used_free_mib(inst.gpu)
                if free >= inst.need_mib + self.headroom_mib:
                    await self._swap(inst, "wake", inst.name)
                    woke = True
                    cond.notify_all()
                    break
                awake = [i for i in self.instances.values() if i.gpu == inst.gpu and i.awake and i is not inst]
                if self.keep:
                    if await self._evict_keep(inst, awake, evict_busy):
                        evictions += 1
                        cond.notify_all()
                        continue
                    try:  # residency expiry is not an event: re-check periodically
                        await asyncio.wait_for(cond.wait(), timeout=0.5)
                    except TimeoutError:
                        pass
                    continue
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

    async def _evict_keep(self, inst, awake, evict_busy=True):
        """Put one instance to sleep with its requests frozen. Returns False if none may go yet."""
        now = time.time()
        movable = [i for i in awake if i.inflight == 0
                   or (evict_busy and now - i.resident_since >= self.min_resident_s)]
        if not movable:
            return False
        victim = min(movable, key=lambda i: i.last_used)
        await self._swap(victim, "sleep", inst.name)
        if victim.inflight:
            asyncio.get_running_loop().create_task(self._rewake(victim))
        return True

    async def _rewake(self, inst):
        """Wake an instance whose requests were frozen by a keep sleep."""
        try:
            # Under --kv-pressure, busy neighbours are evicted only by the pressure rule.
            await self.ensure_awake(inst, evict_busy=not self.kv_pressure)
        except Exception as e:
            print(f"rewake {inst.name} failed: {e}", flush=True)

    async def _queued(self, inst):
        """vllm:num_requests_waiting of an instance (0 if the scrape fails)."""
        try:
            async with self.session.get(f"http://localhost:{inst.port}/metrics",
                                        timeout=aiohttp.ClientTimeout(total=2)) as r:
                text = await r.text()
        except Exception:
            return 0
        for line in text.splitlines():
            if line.startswith("vllm:num_requests_waiting"):
                return int(float(line.rsplit(" ", 1)[1]))
        return 0

    async def kv_pressure_loop(self, gpu):
        """--kv-pressure: round-robin time sharing of a GPU whose instances are short of KV memory."""
        starved_since, last_evict = {}, 0.0
        while True:
            await asyncio.sleep(self.poll_s)
            mine = [i for i in self.instances.values() if i.gpu == gpu]
            awake = [i for i in mine if i.awake]
            queued = dict(zip((i.name for i in awake), await asyncio.gather(*(self._queued(i) for i in awake))))
            now = time.time()
            for i in mine:
                waiting = queued.get(i.name, 0) > 0 if i.awake else i.inflight > 0
                if waiting:
                    starved_since.setdefault(i.name, now)
                else:
                    starved_since.pop(i.name, None)
            starved = [i for i in mine if i.name in starved_since and now - starved_since[i.name] >= self.starve_s]
            busy = [i for i in awake if i.inflight > 0]
            frozen = [i for i in mine if not i.awake and i.inflight > 0]
            _, free = self.used_free_mib(gpu)
            if (not starved or free > self.pressure_free_mib or len(busy) + len(frozen) < 2
                    or now - last_evict < self.pressure_cooldown_s):
                continue
            movable = [i for i in busy if now - i.resident_since >= self.min_resident_s
                       and starved != [i]]
            if not movable:
                continue
            victim = min(movable, key=lambda i: i.last_used)
            reason = "kv_pressure:" + ",".join(
                f"{i.name}={queued.get(i.name, i.inflight)}" for i in starved)
            async with self.cond[gpu]:
                if not victim.awake or victim.inflight == 0:
                    continue
                await self._swap(victim, "sleep", reason)
                self.cond[gpu].notify_all()
            last_evict = time.time()
            asyncio.get_running_loop().create_task(self._rewake(victim))

    async def handle(self, request):
        inst = self.instances.get(request.match_info["name"])
        if inst is None:
            return web.Response(status=404, text="unknown instance")
        body = await request.read()
        rec = {"id": request.headers.get("X-Request-Id"), "instance": inst.name,
               "arrive_s": round(time.time() - self.t0, 4)}
        inst.last_used = time.time()  # LRU: a request arriving or finishing counts as use
        if self.schedule is not None and self.anchor is None:
            self.start_schedule()
        try:
            if self.schedule is not None and not inst.serving and self.pending_wakes.get(inst.name):
                rec["cold"], rec["evictions"] = await self.wait_scheduled_wake(inst)
            else:
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

    # -- replaying a reference run's sleeps and wakes (--schedule) --

    def load_schedule(self, ref_dir):
        """Per GPU: [(seconds after the first request, instance, action)] from a reference run."""
        swaps = [json.loads(l) for l in open(os.path.join(ref_dir, "swaps.jsonl"))]
        reqs = [json.loads(l) for l in open(os.path.join(ref_dir, "proxy_requests.jsonl"))]
        first = min(r["arrive_s"] for r in reqs)
        self.schedule = {}
        self.pending_wakes = {}
        for sw in sorted(swaps, key=lambda x: x["t"]):
            self.schedule.setdefault(sw["gpu"], []).append(
                (sw["t"] - first, sw["instance"], sw["action"], sw.get("inflight", 0)))
            if sw["action"] == "wake":
                self.pending_wakes[sw["instance"]] = self.pending_wakes.get(sw["instance"], 0) + 1
        self.anchor = None

    def start_schedule(self):
        self.anchor = time.time()
        for gpu, acts in self.schedule.items():
            asyncio.get_running_loop().create_task(self._run_schedule(gpu, acts))

    async def _run_schedule(self, gpu, acts):
        cond = self.cond[gpu]
        for t, name, action, ref_inflight in acts:
            delay = self.anchor + t - time.time()
            if delay > 0:
                await asyncio.sleep(delay)
            inst = self.instances[name]
            async with cond:
                if action == "wake":
                    while not inst.awake:
                        _, free = self.used_free_mib(gpu)
                        if free >= inst.need_mib + self.headroom_mib:
                            break
                        try:  # memory frees as requests finish
                            await asyncio.wait_for(cond.wait(), timeout=0.5)
                        except TimeoutError:
                            pass
                    if not inst.awake:
                        await self._swap(inst, "wake", "schedule", self._lag(t))
                    # only now: a request seeing no pending wake falls back to the policy
                    self.pending_wakes[name] -= 1
                elif inst.awake:
                    # The reference slept an idle instance: let this run's stragglers finish
                    # first instead of freezing requests the reference never froze.
                    while ref_inflight == 0 and inst.inflight > 0:
                        try:
                            await asyncio.wait_for(cond.wait(), timeout=0.5)
                        except TimeoutError:
                            pass
                    await self._swap(inst, "sleep", "schedule", self._lag(t))
                cond.notify_all()
        print(f"schedule for GPU {gpu} done", flush=True)
        for inst in self.instances.values():  # safety net: nothing stays frozen after the schedule
            if inst.gpu == gpu and not inst.awake and inst.inflight:
                asyncio.get_running_loop().create_task(self._rewake(inst))

    def _lag(self, t):
        return {"scheduled_s": round(t, 4), "lag_s": round(time.time() - self.anchor - t, 4)}

    async def wait_scheduled_wake(self, inst):
        """Wait for inst's next scheduled wake (falls back to the policy if no wake is left)."""
        cond = self.cond[inst.gpu]
        async with cond:
            while not inst.serving and self.pending_wakes.get(inst.name):
                await cond.wait()
        if not inst.serving:
            return await self.ensure_awake(inst)
        return False, 0

    def _log(self, rec):
        self.reqlog.write(json.dumps(rec) + "\n")
        self.reqlog.flush()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--port", type=int, default=30999)
    ap.add_argument("--headroom-gib", type=float, default=3.0,
                    help="free memory kept beyond the wake size, for KV growth and activations")
    ap.add_argument("--keep", action="store_true",
                    help="evict without draining: sleep with mode=keep (needs the tiered_restore plugin)")
    ap.add_argument("--min-resident-s", type=float, default=5.0,
                    help="with --keep: a woken instance with requests in flight is not evicted before this")
    ap.add_argument("--kv-pressure", action="store_true",
                    help="with --keep: time-share a GPU whose busy instances are short of KV memory")
    ap.add_argument("--pressure-free-gib", type=float, default=6.0,
                    help="KV pressure when NVML free memory is at or below this")
    ap.add_argument("--starve-s", type=float, default=1.0,
                    help="an instance counts as starved after queueing this long")
    ap.add_argument("--pressure-cooldown-s", type=float, default=2.0)
    ap.add_argument("--poll-s", type=float, default=0.5)
    ap.add_argument("--schedule", metavar="REF_RUN_DIR",
                    help="with --keep: replay the sleeps/wakes of this reference run instead of the policy")
    args = ap.parse_args()
    if args.kv_pressure and not args.keep:
        ap.error("--kv-pressure needs --keep")
    if args.schedule and not args.keep:
        ap.error("--schedule needs --keep")
    if args.schedule:
        args.kv_pressure = False

    proxy = Proxy(args.run_dir, int(args.headroom_gib * 1024), args.keep, args.min_resident_s, args.kv_pressure,
                  int(args.pressure_free_gib * 1024), args.starve_s, args.pressure_cooldown_s, args.poll_s)
    # force_close: an instance asleep for a while lets uvicorn close idle keep-alive
    # connections; reusing one fails the next request with ServerDisconnectedError.
    if args.schedule:
        proxy.load_schedule(args.schedule)
    proxy.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600),
                                          connector=aiohttp.TCPConnector(limit=0, force_close=True))
    app = web.Application(client_max_size=64 * MIB)
    app.router.add_get("/health", lambda _: web.Response(text="ok"))
    app.router.add_post("/{name}/{path:.*}", proxy.handle)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, "localhost", args.port).start()
    if args.kv_pressure:
        for gpu in proxy.handles:
            asyncio.get_running_loop().create_task(proxy.kv_pressure_loop(gpu))
    print(f"swap proxy on :{args.port}; wake sizes (MiB): "
          + ", ".join(f"{i.name}={i.wake_mib}" for i in proxy.instances.values()), flush=True)
    await asyncio.Event().wait()


if __name__ == "__main__":
    asyncio.run(main())

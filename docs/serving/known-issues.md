# Known issues serving on Beverin (MI300A)

Problems we hit running LLM endpoints and agent fleets on Beverin, each with the evidence and what we do
about it. Most of them produce no error, so a job looks healthy while its results are wrong. The
per-model pages and [`knobs.md`](knobs.md) carry the configuration that avoids them.

| # | Issue | Visible as | Status |
|---|---|---|---|
| 1 | RCCL internal error on cross-node sends and all-reduces (GLM-5.3, 4 nodes) | server dies after 1-4 h of agent load (tp=4 pp=4), or at graph capture (tp=8 pp=2) | open |
| 2 | A writable CE container freezes on fuse-overlayfs | every agent on the node stalls at once, no error | fixed in hb: the agent container is read-only |
| 3 | vLLM `ROCM_AITER_FA` garbles Qwen3.8 under agent load | random tokens in reasoning, invented tool names | avoided: Qwen3.8 serves on SGLang |
| 4 | Self-tuned GEMM kernel tables break Qwen3.8 numerics | agents loop on degenerate reasoning | avoided: stock aiter tables only |
| 5 | RCCL GPU-direct RDMA corrupts cross-node sums | wrong outputs, no error | avoided: `NCCL_NET_GDR_LEVEL=0` |
| 6 | RCCL silently falls back to TCP | several times slower, correct | check the `Using network` log line |
| 7 | SGLang split groups hang on the RCCL/ROCm build | init hang | avoided: `VLLM_DISTRIBUTED_USE_SPLIT_GROUP=0` |
| 8 | APU out-of-memory kills without a traceback | log stops, exit by signal | fraction rules in [`knobs.md`](knobs.md#the-mi300a-memory-model) |

## 1. GLM-5.3: RCCL `ncclInternalError` on the pipeline send

**Setup.** `zai-org/GLM-5.3` fp8 on SGLang, `tp=4 pp=4` over four MI300A nodes, RCCL over Slingshot through
the `aws-ofi-nccl` plugin (`NCCL_NET="AWS Libfabric"`, `FI_PROVIDER=cxi`, `FI_CXI_RX_MATCH_MODE=software`,
`FI_MR_CACHE_MONITOR=disabled`, `NCCL_NET_GDR_LEVEL=0`). 12 to 48 concurrent agent conversations.

**What happens.** After 1 to 4 hours of steady decode, one pipeline stage's scheduler raises during the
send of a step's tensors to the next stage:

```
scheduler_pp_mixin.py, in _pp_send_dict_to_next_stage
torch.distributed.DistBackendError: NCCL error in: .../NCCLUtils.cpp:93, internal error - please report this
issue to the NCCL developers, NCCL version 2.27.7
ncclInternalError: Internal check failed.
```

The other stages then fail with `gloo ... Connection closed by peer`, and teardown logs
`Failed to destroy CXI Service ID 5 (cxi0): Device or resource busy` on every node. It happened in all four
GLM-5.3 jobs of 2026-10-10/11 (675850 after 3 h 07 min, 675956 after 2 h 30 min, 675958 after 18 min of
serving, 675957 ended in `NODE_FAIL`). Earlier GLM-5.3 jobs on the same shape served for up to 14 h (633828).
Nothing in our stack changed between them, and the error comes from RCCL's own internal check, not from SGLang.

**A second shape fails the same way.** On 4 nodes as `tp=8 pp=2` (`SGLANG_PP_SIZE=2`, smoke 676465), each
tensor-parallel group spans two nodes, and the server died before serving a request: the cross-node all-reduce
raised the same RCCL internal error while SGLang captured CUDA graphs. (A first attempt, 676429, failed earlier:
SGLang's post-load barrier is hardcoded to 480 s and the two nodes of a group finished loading weights up to
18 minutes apart; the sglang image now reads `SGLANG_UNBALANCED_MODEL_LOADING_TIMEOUT_S`.) So both a pipeline send
and a tensor-parallel all-reduce fail inside RCCL whenever they cross nodes through the plugin.

**Open.** We have not yet captured RCCL's own diagnostic (`NCCL_DEBUG=WARN` set for 676465 printed no RCCL
lines into the server logs). Advice from CSCS is welcome on: CXI queue sizes (`FI_CXI_DEFAULT_CQ_SIZE`,
`FI_CXI_DEFAULT_TX_SIZE`), the netstack artifact `gpu_rocm7-cxi_13.1.0-ofi_2.6.0-aws_1.20.0` (26.08.1) with
RCCL 2.27.7, and any known RCCL/libfabric issue with long-lived cross-node traffic on MI300A.

## 2. A writable container freezes on fuse-overlayfs

**Setup.** One agent node runs up to 60 agents in one CE container step. With `writable = true` the CE stacks
a fuse-overlayfs upper layer over the squashfs image.

**What happens.** The container stops making progress all at once. Processes already streaming keep writing
to their bind mounts, but no new process starts: the node monitor stops sampling, no agent issues another
request, and nothing reports an error. In temperature3 job 671243 this lasted 7 h. In job 675636 (Qwen3.8 at
temperature 1.5, whose agents issue commands such as `rm -rf /root`) the agent node's monitor and its last
request both stop at 21:51:49, and the job's stall watchdog stopped the run an hour later. A copy-up of a large
image tree through fuse-overlayfs is consistent with every symptom. The later
`Transport endpoint is not connected` on `/opt/venv` is the teardown unmounting the image under exiting
processes, not the cause.

**What we do.** The agent container runs read-only (`writable = false`, set by `derived_edf` in
`hpcagent_bench/cluster/container_runtime.sh`), with the node's own `/tmp` bound in. Agents write only to bind
mounts and to the per-agent tmpfs the seal gives each one, so a stray write fails at once with `EROFS`. A run
that still sees no change under its agent and judge folders for `AGENT_STALL_ABORT_SECONDS` (default 3600) is
stopped and its unfinished kernels are rerun.

## 3. vLLM with `ROCM_AITER_FA` garbles Qwen3.8 under agent load

vLLM 0.28 with aiter flash attention (`ROCM_AITER_FA`, `VLLM_ROCM_USE_AITER=1`) passed every synthetic gate
(300 to 500 output tokens), but under real agent load 65-75% of the Qwen3.8 agents produced random CJK text in
their reasoning and called tools that do not exist. gpt-oss-120b on the same image stayed clean. Qwen3.8 serves on
SGLang instead. Short coherence checks did not catch it (1 of 40 conversations at 4 turns), so an engine
switch now needs a long multi-turn check, never a throughput number alone.

## 4. Self-tuned kernel tables break Qwen3.8

aiter ships tuned bf16 GEMM tables for gfx950 and gfx1250 only, so on MI300A (gfx942) several shapes run hipBLASLt.
We generated our own MI300A table for speed. It selected split-K kernels that accumulate in bf16, which put up
to 3.75% of the elements of a GEMM outside tolerance, and Qwen3.8 agents served under it ended in degenerate
reasoning loops. We serve with the stock tables and accept the slower fallbacks
([`knobs.md`](knobs.md#aiter-correctness-first-fallbacks-accepted)). Speed tuning of kernels must be checked
for numerical tolerance against a reference, not only for throughput.

## 5-8. Fabric and memory

- **GPU-direct RDMA.** RCCL enables it by itself, and on this fabric it silently corrupts cross-node sums.
  Set `NCCL_NET_GDR_LEVEL=0` (not `NCCL_GDR_LEVEL`).
- **TCP fallback.** Without the plugin RCCL runs over TCP: correct, several times slower, no error.
  `grep -a "Using network" server-0.log` must say `AWS Libfabric`; our EDFs set `NCCL_NET_PLUGIN=ofi` so a
  missing plugin is an error instead.
- **Split groups.** Any SGLang/vLLM option that enables split process groups hangs init on this RCCL build.
- **APU memory.** GPUs and host share one pool, `--mem-fraction-static` is node-wide, and an out-of-memory death
  is the host OOM killer: the log stops with no traceback.

## Model behaviour that is not infrastructure

At sampling temperature 1.5 both Qwen3.8 and gpt-oss-120b degrade: Qwen3.8 emits long runs of random tokens
inside otherwise valid tool calls, and gpt-oss-120b emits malformed harmony tool headers, so its episodes end after
one turn. That is the model, not the server; the temperature3 study measures it.

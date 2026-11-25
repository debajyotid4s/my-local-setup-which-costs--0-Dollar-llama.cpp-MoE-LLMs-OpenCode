# Local agentic coding with llama.cpp + OpenCode

Zero-API-cost coding agent: a local LLM served by `llama-server`, driven by **OpenCode v2.3.0**
(V2 config and plugin API). Two launchers are included: `qwen3.6-35b-a3b` and
`ornith-1.5-35b-a3b`. Target hardware: one NVIDIA GPU (developed on a 16 GB Tesla V100).

## #Disclaimer: The Qwen 3.6-35B-a3b related launching scripts are dropped from this repo, as i am totally shifted to Ornith, it is hard to maintain two LLMs regularly on my poor machine. So as far as I used, the information are given here, if you want to use it, please research the latest HuggingFace card of the model, and change the configurations as needed.

## Layout

```
scripts/
  01-install-deps.sh              packages, hf CLI, OpenCode
  02-build-llama-cpp.sh           llama.cpp build (set CMAKE_CUDA_ARCHITECTURES)
  03-download-model.sh            GGUF download (MODEL_REPO, MODEL_QUANT)
  04-install-opencode-config.sh   deploy config/ to ~/.config/opencode/
  05-init-project.sh <dir>        per-project .agent/ memory files
  <model>/llm-up.sh               start llama-server
  <model>/llm-stop.sh             graceful shutdown (SIGTERM -> 30s -> SIGKILL)
config/
  opencode.json                   -> ~/.config/opencode/opencode.json
  AGENTS.global.md                -> ~/.config/opencode/AGENTS.md
  commands/handoff.md             -> ~/.config/opencode/commands/handoff.md
  plugins/deferred-compaction/    -> ~/.config/opencode/plugins/deferred-compaction/
  project-templates/              copied into projects by 05-init-project.sh
```

## Hardware

| GPU | `CMAKE_CUDA_ARCHITECTURES` | CUDA |
|---|---|---|
| Tesla V100 / Quadro RTX 4000–6000 | `70` | **12.x only** (CUDA 13 dropped Volta) |
| GTX 10xx (Pascal) | `61` | 12.x only |
| RTX 20xx / T4 | `75` | 12.x or 13.x |
| RTX 30xx / A100 | `86` | 12.x or 13.x |
| RTX 40xx | `89` | 12.x or 13.x |
| RTX 50xx / H100 | `90` | 12.x or 13.x |

Models (MoE, hybrid CPU/GPU via `--n-cpu-moe`):

| Model | Quant | Disk | `LLM_NCMOE` start |
|---|---|---|---|
| Qwen3.6-35B-A3B | UD-IQ4_XS | ~17.7 GB | 18 |
| Ornith-1.5-35B-A3B | IQ4_XS | ~19.3 GB | 18 |

System RAM: 32 GB+ is comfortable; 16 GB needs a higher `LLM_NCMOE` and no `LLM_MAX_MEM`/`--mlock`.

## Setup

```bash
cd qwen-v100-opencode
chmod +x scripts/*.sh
./scripts/01-install-deps.sh                                  # SKIP_OPENCODE=1 to skip
CMAKE_CUDA_ARCHITECTURES=70 ./scripts/02-build-llama-cpp.sh   # LLAMA_REF=<tag> to pin
MODEL_REPO=<repo> MODEL_QUANT=UD-IQ4_XS ./scripts/03-download-model.sh
./scripts/04-install-opencode-config.sh
./scripts/05-init-project.sh /path/to/project
```

Start a server in its own terminal (do not Ctrl+C it):

```bash
./scripts/ornith-1.5-35b-a3b/llm-up.sh 2>&1 | tee -a ~/llm-server.log
curl -s http://127.0.0.1:8080/health
```

Then run `opencode` in the project directory. OpenCode installs its VS Code extension automatically.

Verify provider/model/context: `opencode api get /api/model/default`.

## Tuning

All knobs are environment variables on `llm-up.sh`.

| Variable | Default | Effect |
|---|---|---|
| `LLM_CTX` | `262144` | Context window; ~1.3 GB VRAM saved at `131072` |
| `LLM_NCMOE` | `18` | MoE layers whose experts stay in system RAM; lower = more VRAM used |
| `LLM_UBATCH` / `LLM_BATCH` | 128–512 | Micro-batch; costs VRAM, helps long prefill |
| `LLM_THREADS` | physical cores | Decode threads |
| `LLM_KV_K` / `LLM_KV_V` | `q8_0` / `q8_0` | KV cache quant; `q4_0` V saves VRAM |
| `LLM_COMPACTION_AT` | `75% of LLM_CTX` | Client threshold hint; auto-tracks `LLM_CTX` overrides |
| `LLM_MOE_CACHE_MIB` | `1024` | Ornith only: drop to 512/0 on CUDA OOM |
| `LLM_THINKING` | `1` | `1` = thinking/default, `0` = non-thinking; determines per-mode sampling defaults |
| `LLM_TEMP` / `LLM_TOP_P` / `LLM_TOP_K` / `LLM_MIN_P` | mode-dependent (see below) | Sampling; any can be overridden individually |
| `LLM_MODEL` / `LLM_SERVER` / `LLM_PORT` | — | Override GGUF, binary, port |

Qwen launcher only (16 GB RAM machines should keep these off): `LLM_MAX_MEM` (cgroup cap),
`LLM_MLOCK`, `LLM_PROMPT_CACHE`, `LLM_SWAP_CHECK`, `LLM_REASONING_BUDGET`.
Mixed K/V types need `-DGGML_CUDA_FA_ALL_QUANTS=ON`.

Never set OpenCode's model `limit.context` above the server's `LLM_CTX`.

## Ornith launcher changes

The Ornith launcher (`ornith-1.5-35b-a3b/llm-up.sh`) was revised with three substantive
changes. The Qwen launcher is unaffected.

### Sampling defaults are now mode-dependent

Previously the launcher used a single set of sampling parameters regardless of `LLM_THINKING`.
The Qwen3 model card prescribes different values per mode:

| Mode | `LLM_TEMP` | `LLM_TOP_P` | `LLM_TOP_K` | `LLM_MIN_P` |
|---|---|---|---|---|
| Thinking (`LLM_THINKING=1`, default) | `0.6` | `0.95` | `20` | `0.0` |
| Non-thinking (`LLM_THINKING=0`) | `0.7` | `0.8` | `20` | `0.0` |

The original launcher applied the thinking-mode values in both cases; with `LLM_THINKING=0` the
server was running non-thinking mode under a tighter temperature but a wider `top_p` than
intended. Any of the four variables can still be overridden individually via their `LLM_*`
equivalents regardless of mode.

**Tradeoff:** Non-thinking mode's `top_p=0.8` is tighter than the former `0.95`. It narrows
the candidate distribution at each step, which reduces variety slightly but also cuts more
low-probability tokens. If you want broader sampling in non-thinking mode (e.g., for creative
tasks), set `LLM_TOP_P=0.95` explicitly. For agentic coding sessions, the tighter default is
the more conservative choice.

### `--no-context-shift` is now a hard requirement

Previously, if the llama-server binary lacked this flag, the launcher emitted a warning and
continued. It now calls `fail()` and exits before launch.

**Why:** The deferred-compaction plugin counts on the server returning an error when the context
fills rather than silently truncating old tokens. With context-shift enabled, the server
discards early conversation history without surfacing any signal to the client — the agent
continues generating from a corrupted view of the session. That is not a soft performance
degradation; it is a silent correctness failure that is difficult to diagnose after the fact.

**Tradeoff:** None in practice. `--no-context-shift` has been in llama.cpp since well before
commit 77e2db8, which is already stated as the build minimum for Ornith's SM70 FA kernels. Any
compliant build already has the flag.

### `COMPACTION_AT` now auto-tracks `LLM_CTX`

The default was previously hardcoded to `196608`. It is now computed as `LLM_CTX × 0.75` at
launch time.

When `LLM_CTX=262144` (the unchanged default) the result is exactly `196608` — identical to
before. When `LLM_CTX` is overridden, the old hardcoded value could exceed the entire context
window (e.g., `LLM_CTX=65536` → old default `196608 > 65536`), making the launch summary show
a threshold that would never fire. The derived default is always consistent with the active
window and with what the deferred-compaction plugin computes internally (0.75 of `LLM_CTX`).

**Tradeoff:** None. `LLM_COMPACTION_AT` can still be set explicitly to any value below
`LLM_CTX` to override.

## Compaction

`compaction.auto: false`. The `deferred-compaction` plugin compacts only at a safe point:
execution finished, no tool in flight, last step `stop`, session idle.

- Pending at **75%** of the window; proactive at **65%** after 10 min idle.
- Keeps ~15,000 recent tokens (`compaction.keep.tokens`); the session continues, no `/new`.
- One compaction at a time; manages the local `v100` provider only (`*` = all providers).
- `/compact` still works manually; `/handoff` + `/new` only for a deliberate reset.

See `config/plugins/deferred-compaction/README.md` for the full behaviour and tests.

## Memory safety and warm start

If the machine reboots under load, the OOM killer is at work:

```bash
LLM_MAX_MEM=10G ./scripts/qwen3.6-35b-a3b/llm-up.sh   # cap (qwen launcher)
LLM_CTX=131072 ./scripts/<model>/llm-up.sh            # less KV cache
```

Add 8 GB swap as a safety net, and raise `LLM_NCMOE` if VRAM is short.
`LLM_MLOCK=1` requires `LLM_MAX_MEM=` (empty) and enough RAM for the whole model.

Warm start (qwen launcher): `LLM_PROMPT_CACHE=1` saves the KV cache to disk and restores it on
restart (~5–15 s, ~2–4 GB disk). Verify with `grep "forcing full prompt re-processing" ~/llm-server.log`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `no kernel image is available` | Wrong CUDA arch; rebuild with the right `CMAKE_CUDA_ARCHITECTURES` |
| CUDA OOM at load | Raise `LLM_NCMOE`, lower `LLM_CTX`/`LLM_UBATCH`/`LLM_MOE_CACHE_MIB` |
| Low GPU utilisation | Expected: experts are on CPU. Lower `LLM_NCMOE` |
| Every turn re-reads history | Prompt cache disabled or invalidated; do not edit `HANDOFF.md` mid-session |
| A long session stops compacting after a restart | State is in-memory: run `/compact` once on a resumed session near the limit |
| OpenCode request dies after ~5 min | `headerTimeout`/`chunkTimeout` must be `false` |
| Tool calls arrive as plain text | Ensure `--jinja`; re-download the GGUF |
| Port 8080 busy | `./scripts/<model>/llm-stop.sh`, or set `LLM_PORT` and update `baseURL` |
| `hf: command not found` | `export PATH="$HOME/.local/bin:$PATH"` |
| `Bad Request` / repeated "go on" | Context overflow or damaged history: check `~/llm-server.log`, then `/undo`, `/compact` |
| Config ignored | `opencode --version` must be 2.x |
| Suspend hangs | NVIDIA UVM deadlock: `sudo systemctl mask nvidia-suspend.service` |

## Docker

The image ships only `llama-server`. OpenCode stays on the host and talks to
`http://127.0.0.1:8080/v1`. Models are mounted read-only; nothing is baked in.

```bash
docker build --build-arg CMAKE_CUDA_ARCHITECTURES=70 -t local-llm-opencode:latest .

# default CMD: Qwen3.6-35B-A3B, 256K
docker run -d --name llm-server --gpus all -p 127.0.0.1:8080:8080 \
  -v ~/models/qwen3.6:/models/qwen3.6:ro local-llm-opencode:latest

# Ornith-1.5-35B-A3B, non-thinking mode (temp/top_p per Qwen3 model card for --reasoning off)
docker run -d --name llm-server --gpus all -p 127.0.0.1:8080:8080 \
  -v ~/models/ornith1.5:/models/ornith1.5:ro local-llm-opencode:latest \
  -m /models/ornith1.5/Ornith-1.5-35B-A3B-IQ4_XS.gguf \
  --alias ornith-1.5-35b-a3b-iq4-xs -ngl auto --n-cpu-moe 18 --fit off \
  -c 262144 -np 1 -fa on -ctk q8_0 -ctv q8_0 -b 2048 -ub 512 --jinja \
  --reasoning off --chat-template-kwargs '{"enable_thinking":false}' \
  --no-context-shift \
  --temp 0.7 --top-p 0.8 --top-k 20 --min-p 0.0 --host 127.0.0.1 --port 8080

docker stop llm-server     # graceful; docker kill to force
```

`LLAMA_REF` is pinned to `b12375` (current release; 77e2db8 is the minimum for Ornith's
SM70 FA kernels). Mixed K/V types need `-DGGML_CUDA_FA_ALL_QUANTS=ON` at build time.
Memory cap on a 16 GB RAM host: add `--memory 12g` to `docker run`. The Ornith launcher
deliberately keeps mmap and applies no cgroup cap; the Qwen launcher defaults to
`LLM_MAX_MEM=12G`.

## Daily workflow

1. Start the server (dedicated terminal), then `opencode` in the project.
2. Work; the agent appends milestones to `.agent/SCRATCH.md`.
3. Compaction happens automatically at a safe point; the session continues.
4. Stop the server with `./scripts/<model>/llm-stop.sh` (never Ctrl+C).

Memory files per project: `.agent/HANDOFF.md` (long-term, auto-loaded, rewrite — max 60 lines)
and `.agent/SCRATCH.md` (journal, read with `tail -n 40`). `.agent/` is git-excluded.

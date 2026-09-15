# Jarvis

A staged build of a personal, laptop-hosted assistant. **Component 0 is in
progress. This is not yet a 24/7 daemon and cannot control your computer.**

## Build order and architecture

Only implement and test one component at a time. Do not add inert scaffolding
for later features. The repository root is the `jarvis/` directory in the brief.

| Stage | Module | Responsibility | Status |
| --- | --- | --- | --- |
| 0 | `orchestrator/`, `main.py` | NIM provider, registry, tool loop, sessions | In progress |
| 1 | `bridge/discord.py` | Owner-only Discord transport, attachments, status | Not built |
| 2 | `tools/pc_control.py` | Accessibility verification and explicit approvals | Not built |
| 3 | `extension/`, `bridge/browser.py` | Authenticated local DOM bridge | Not built |
| 4 | `tools/screen_vision.py` | Opt-in, privacy-filtered vision fallback | Not built |
| 5 | `memory/` | Git-logged graph, fuzzy fallback, reminders | Not built |
| 6 | `voice/` | Wake word, STT, multilingual TTS, persistent mode | Not built |
| 7 | `hud/` | Native Qt overlay and live captions | Not built |
| 8 | `main.py`, `service/` | Supervision and Windows interactive-session startup | Not built |

Cross-cutting rules: no secrets in git; deny sensitive actions unless explicitly
approved by an authenticated owner; treat retrieved material as untrusted data;
block private apps/domains before reading or transmitting content; opt-in screen
capture only. Future adapters must call the same orchestrator and register tools,
not implement separate model loops.

## NVIDIA verification (2026-09-15)

- [Hosted LLM API reference](https://docs.api.nvidia.com/nim/reference/llm-apis)
  lists `https://integrate.api.nvidia.com/v1/chat/completions` and
  `qwen/qwen3-next-80b-a3b-instruct`.
- [Qwen3-Next model card](https://docs.api.nvidia.com/nim/reference/qwen-qwen3-next-80b-a3b-instruct)
  explicitly describes tool calling and agent use. The generated inference docs
  do not expose all tool-schema fields. A real account-level tool-call probe is
  still required: a model card is not an endpoint compatibility test.
- [NVIDIA FAQ](https://docs.api.nvidia.com/nim/docs/product) describes free API
  endpoints for **prototyping**. This is not an unlimited-capacity or production
  availability promise. Review the applicable API Trial Terms and account terms
  before relying on this for a daily assistant.
- A universal current free-tier RPM/credit allowance was **not verified** in the
  official API documentation. Forum reports of 40 RPM are not a service contract.
  Local request pacing is a conservative app policy, not a claimed NVIDIA limit.
- There is no paid-provider fallback. Free eligibility, future pricing, model
  availability, and $0 operation indefinitely cannot be guaranteed by this repo.
- Roman Urdu comprehension and Urdu-script output still need evaluation with the
  user's examples; multilingual model support is not proof of speech quality.

## Development tooling

GitNexus is a development aid, not a runtime dependency or personal-memory store.
After each component, run `npx gitnexus analyze` and run `npx gitnexus setup` in
your own coding environment to wire its MCP integration. Never index `.env`,
private session data, or screenshots. Do not commit generated databases.

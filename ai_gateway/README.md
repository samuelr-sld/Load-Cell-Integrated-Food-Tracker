# PlateGo local AI gateway

A small authenticated HTTP API that lets the PlateGo app on a phone use an Ollama model on this laptop over
local Wi-Fi. No internet is needed once the model is installed. It is separate from the camera testbench
(port 8000) and uses port 8100.

```
iPhone (Expo Go)                              Laptop
 PlateGo app ── HTTP + pairing token ──▶  ai_gateway :8100 ── loopback ──▶ Ollama :11434
 ├ runs the 6 nutrition tools on its own SQLite diary
 ├ sends only the question + small tool-result summaries
 └ validates every tool call, number and chart itself
```

## Start (laptop)

From the repository root (`Electives 2`), with the Ollama app running:

```powershell
ollama list                                   # the model must be listed (default: qwen2.5-coder:3b)
.\venv\Scripts\python.exe -m ai_gateway       # binds to the Wi-Fi address on port 8100
```

It prints the **address** and **pairing token** to type into the app, and loads the model so the first
question is fast. The first time it listens on Wi-Fi, Windows Firewall asks for permission: allow **Private
networks** only.

Options (environment variables):

| Variable | Default | Meaning |
|---|---|---|
| `--host` / `--port` | Wi-Fi address / 8100 | interface to bind (`0.0.0.0` is refused) |
| `PLATEGO_AI_MODEL` | `qwen2.5-coder:3b` | Ollama model (`ollama pull <name>` first) |
| `PLATEGO_AI_TOKEN` | random, kept in `ai_gateway/.pairing_token` | pairing token (16+ chars) |
| `PLATEGO_OLLAMA_URL` | `http://127.0.0.1:11434` | must be loopback |
| `PLATEGO_AI_TIMEOUT` | 90 | seconds per model call |
| `PLATEGO_AI_CONCURRENCY` | 1 | simultaneous model calls (more get `429 busy`) |

To issue a new token, delete `ai_gateway/.pairing_token` and restart.

## Connect (iPhone)

1. Phone on the same Wi-Fi as the laptop. Start the app as usual (`npx expo start` in `DietaryApp`, open in Expo Go).
2. Tap **💬 Ask the nutrition assistant**, then **Settings**. Choose **Laptop AI**, enter the printed address
   (for example `http://192.168.1.8:8100`) and token, then tap **Save & test**. The dot turns green.
3. If iOS asks whether Expo Go may find devices on your local network, allow it.

## API

All calls need `Authorization: Bearer <token>`. There is no other route: no model listing, pulling,
deleting or raw Ollama access, and no API docs page.

- `GET /v1/health` returns `{gateway, ollama: "ok"|"unavailable", model, modelInstalled, busy}`.
- `POST /v1/chat` takes `{requestId, messages: [{role: system|user|assistant|tool, content}], format?: "json"}`
  (at most 32 messages, 16,000 characters each, 64 KB in total; unknown fields rejected) and returns
  `{requestId, model, content, timings}`. The model, options and context size are fixed by the gateway.
  If the phone disconnects, generation is cancelled.
- Errors return `{error, message}`: 401 `unauthorized`, 413 `too_large`, 422 `invalid_request`, 429 `busy`,
  503 `ollama_unavailable` / `model_missing`, 504 `timeout`, 502 `ollama_error`.

## Security limits (development demo)

- Traffic is **plain HTTP**: anyone on the same Wi-Fi can read the questions, summaries and the token.
  Use only on a trusted home or lab network, never on public Wi-Fi.
- The token is a shared secret, stored in plain text on the laptop and in the app's AsyncStorage.
  It is not per-user or per-device, and it never expires.
- The gateway does no rate limiting beyond one call at a time. It cannot reach the phone's data. The worst a
  token holder can do is make the model generate text.

## Tests and benchmark

```powershell
.\venv\Scripts\python.exe -m pytest ai_gateway -q                        # gateway contract (fake Ollama)
cd DietaryApp; node ai/benchmark.mjs http://127.0.0.1:8100 <token> --cold  # live model: latency, tool choice, grounding
```

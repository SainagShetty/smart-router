# ops — monitoring & alerts

`healthcheck.sh` is a watchdog for the running smartrouter server. Run it on a schedule
(launchd on macOS, cron on Linux). It emails on the **transition** into a bad state and
again on recovery — no repeat spam while something stays down.

## What it checks

| Check | Alert when | Why it matters |
|-------|-----------|----------------|
| `GET /health` | non-200 / not `ok` | the shared backend is down → every dependent service fails |
| `/health` latency | > `LATENCY_THRESHOLD_MS` (2000) | degradation |
| PM2 process | not `online`, or restart count jumps ≥ `RESTART_JUMP` (3) | crash / restart loop |
| Ollama `/api/tags` | unreachable | local tier down → requests fall back to paid cloud |

## Config (env, with defaults)

`SMARTROUTER_URL` (`http://127.0.0.1:4000`), `SMARTROUTER_PM2_NAME` (`smart-router`),
`OLLAMA_URL` (`http://localhost:11434`), `LATENCY_THRESHOLD_MS` (`2000`), `RESTART_JUMP` (`3`).

Email delivery uses `~/.alert_credentials` (iCloud SMTP) and sends to the address configured
there. State (for de-duping) lives in `ops/.alert_state/`; logs in `ops/healthcheck.log`
(both git-ignored).

## Install (macOS launchd, every 5 min)

```bash
cp ops/com.smartrouter.healthcheck.plist.example \
   ~/Library/LaunchAgents/com.smartrouter.healthcheck.plist
# edit the two /ABSOLUTE/PATH/TO/ placeholders to this repo's path
launchctl load -w ~/Library/LaunchAgents/com.smartrouter.healthcheck.plist
```

Disable: `launchctl unload ~/Library/LaunchAgents/com.smartrouter.healthcheck.plist`.
Run once by hand: `bash ops/healthcheck.sh` (prints status; emails only on a bad transition).

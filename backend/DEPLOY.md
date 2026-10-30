# Deploying the reviewer (GitHub App backend)

Architecture (design doc Module 4, decided 2026-10-23):

```
GitHub ──webhook──▶ Caddy (HTTPS) ─▶ api (FastAPI) ─▶ Redis ─▶ worker (Celery)
                         GCP VM, docker compose          │          │
                                                   Postgres ◀───────┤
                                                                    ▼  Tailscale
                                                     model API on the DGX Spark (prana.api)
```

The VM needs no GPU. The model stays on the Spark and is reached only over
Tailscale.

**Two ways to host it.** Option A (below, current choice since 2026-10-30:
there are no cloud credits) runs the whole stack on the Spark and publishes
it with Tailscale Funnel. Option B is a cloud VM (steps 2-4 further down).
Step 1 is the same for both.

## Option A — on the DGX Spark, published with Tailscale Funnel (no cost)

```
GitHub ─https─▶ Tailscale Funnel ─▶ 127.0.0.1:8080 Caddy ─▶ api ─▶ Redis ─▶ worker ─▶ model API
                (spark-5b23.tail75b36b.ts.net, real certificate)               (same machine)
```

Nothing listens on the campus LAN; Funnel reaches the Spark through
Tailscale, so no router or firewall change is needed. Verified 2026-10-30
with throwaway secrets: all 7 services healthy, routes correct, `/api` 404,
dashboard 401 without the password, not reachable on the LAN address, the
worker container reaches the model API, signed webhook queued in 153 ms.

One-time, done by you (needs the tailnet admin and sudo on the Spark):

1. Tailscale admin console → **DNS**: enable **HTTPS certificates**; then
   **Access controls**: allow Funnel for the Spark (the console offers an
   "Enable Funnel" button that adds the `funnel` node attribute).
2. On the Spark: `sudo tailscale set --operator=$USER` so Funnel can be
   managed without sudo afterwards.

Then:

```bash
cd ~/prana/backend
mkdir -p secrets && cd secrets          # the same files as step 3 below
#   ... github_private_key.pem, github_webhook_secret, model_api_key,
#       postgres_password, database_url, dashboard_token
sudo chown 10001:10001 * && sudo chmod 400 * && sudo chown 70:70 postgres_password
cd .. && cp .env.example .env           # PUBLIC_HOST=spark-5b23.tail75b36b.ts.net,
                                        # GITHUB_APP_ID, DASHBOARD_* (step 3)
docker compose -f docker-compose.yml -f docker-compose.spark.yml up -d --build
docker compose -f docker-compose.yml -f docker-compose.spark.yml ps   # all healthy
tailscale funnel --bg 8080
curl -s https://spark-5b23.tail75b36b.ts.net/health                     # {"ok":true}
```

GitHub App webhook URL: `https://spark-5b23.tail75b36b.ts.net/webhook`.
Dashboard: `https://spark-5b23.tail75b36b.ts.net/dashboard/`.
Stop publishing at any time with `tailscale funnel --bg 8080 off`.

Trade-offs to know: availability is the Spark's (college power and network);
the stack shares the machine with the ERP agent, but it uses CPU and a few
hundred MB of memory, never the GPU; and every `docker compose` command for
this deployment needs both `-f` files.

## 1. Register the GitHub App (~10 min, on github.com)

Settings → Developer settings → GitHub Apps → New GitHub App.

| Field | Value |
|---|---|
| Webhook URL | Option A: `https://spark-5b23.tail75b36b.ts.net/webhook` · Option B: `https://<PUBLIC_HOST>/webhook` |
| Webhook secret | `openssl rand -hex 32` - keep it for step 3 |
| Repository permissions | Pull requests: **Read and write** · Contents: **Read** · Metadata: **Read** |
| Subscribe to events | **Pull request** |
| Where can it be installed | Only on this account (until the demo is public) |

After creating it: note the **App ID**, then "Generate a private key" (a
`.pem` downloads). Install the App on the demo repository.

## Option B — a cloud VM

## 2. Create the VM (~15 min, GCP console or gcloud)

- e2-standard-2, `asia-south1`, Ubuntu 24.04 LTS, 30 GB disk, a static
  external IP; firewall: allow 80 and 443 only (SSH via IAP or your IP).
- DNS: an A record for `PUBLIC_HOST` pointing at the static IP. With no
  domain, `<ip-with-dashes>.sslip.io` works and Caddy gets a real certificate.
- On the VM:

```bash
curl -fsSL https://get.docker.com | sh && sudo usermod -aG docker $USER
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up            # then, in the Tailscale admin console, share ONLY the
                             # spark-5b23 node with this machine - not the whole tailnet
curl -s http://100.124.121.97:8100/health     # the model API answers
```

## 3. Secrets (files, never env values; never committed)

```bash
git clone <repo> && cd <repo>/backend && mkdir -p secrets && cd secrets
cp ~/Downloads/<app>.private-key.pem github_private_key.pem
printf '%s' '<webhook secret from step 1>' > github_webhook_secret
printf '%s' '<model API key, from the Spark: ~/.config/prana/api.key>' > model_api_key
openssl rand -hex 24 > postgres_password
printf 'postgresql+psycopg://prana:%s@postgres:5432/prana' "$(cat postgres_password)" > database_url
openssl rand -hex 24 > dashboard_token
# The app runs as uid 10001 inside the container and must be able to read them:
sudo chown 10001:10001 * && sudo chmod 400 *
sudo chown 70:70 postgres_password         # postgres (alpine) reads its own
cd .. && cp .env.example .env && nano .env  # PUBLIC_HOST, GITHUB_APP_ID, DASHBOARD_*
# Dashboard login: make the hash, then paste it into .env with every "$" doubled:
docker run --rm caddy:2.8-alpine caddy hash-password --plaintext '<choose a password>'
```

## 4. Start and verify

```bash
docker compose up -d --build
docker compose ps                          # all healthy
curl -s https://$PUBLIC_HOST/health        # {"ok":true}
```

- GitHub App settings → Advanced → Recent deliveries: the `ping` shows **200**.
- Open a PR on the demo repo touching a `.py`/`.ts`/`.js`/`.go`/`.rs` file:
  a review with inline comments appears (milestone M4: within 30 s).
- Dashboard: `https://$PUBLIC_HOST/dashboard/` (user `DASHBOARD_USER`, your password).
  Reviews, latency against the 30 s target, severity and issue types,
  confidence, and 👍/👎 feedback (collected from GitHub every 15 minutes).
- The JSON API (`/api/*`) is deliberately not published; only `/webhook`,
  `/health` and `/dashboard` are reachable from outside.

## Operations

| Task | Command |
|---|---|
| Logs | `docker compose logs -f api worker` |
| Failed runs (dead letters) | `docker compose exec worker python -m prana_app.admin failed` |
| Replay one | `docker compose exec worker python -m prana_app.admin replay <run_id>` |
| Backup | `docker compose exec postgres pg_dump -U prana prana \| gzip > backup-$(date +%F).sql.gz` (cron daily) |
| Update | `git pull && docker compose up -d --build` - CI has already run the tests |
| Roll back | `git checkout <previous tag> && docker compose up -d --build` |
| Rotate the webhook secret | new value in GitHub + `secrets/github_webhook_secret`, then `docker compose restart api` |

## The model side (DGX Spark)

The model API must be up for reviews to run; the worker retries three times
with back-off and then posts a "could not review" comment instead of going
silent. Today it runs as the GPU-queue job `prana-api`; for production it
should become a systemd service so it restarts on failure and on boot
(needs sudo on the Spark - see `scripts/serve_api.sh` for the command).

## Schema changes

Tables are created at start-up (`create_all`), which adds missing tables but
never alters existing ones. The first deployment starts empty, so this is
enough today; the first change to an existing table after go-live needs a
migration (Alembic) instead.

## Smoke test without GitHub

Full stack, 2026-10-30, throwaway secrets, torn down after: all seven
services healthy; Caddy served `/health` 200, `/api/*` 404, `/dashboard`
401 without or with a wrong password and 200 with the right one, HSTS and
nosniff headers set, no Server header; the dashboard rendered 18 seeded runs
(KPIs, charts, tables) in headless Edge; the feedback task swept 16 PRs and
logged each GitHub refusal without failing.

Earlier run (backend only):

Verified 2026-10-30 on the Spark with throwaway secrets: signed webhook →
202 in 112 ms; bad signature → 401; redelivery → ignored; the worker picked
the job up, GitHub refused the fake App (404), the run was recorded as failed
and the dashboard API reported it; no token → 401.

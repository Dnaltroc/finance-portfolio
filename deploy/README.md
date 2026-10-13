# Docker and Kubernetes

The Streamlit app runs as a container image and deploys to Kubernetes. It is the same app as the [hosted version](https://finance-portfolio-dnaltroc.streamlit.app/); nothing new is added on the backend.

## Resume line

> Containerized a Python/Streamlit analytics app (non-root, read-only filesystem) and deployed it to Kubernetes with health probes, zero-downtime rolling updates and secrets; CI builds the image, deploys it to a throwaway kind cluster, renders every page inside the pod, and publishes to GitHub Container Registry.

## What is here

| File | What it does |
| --- | --- |
| `Dockerfile` | `python:3.12-slim`; installs dependencies in their own layer so code edits rebuild in seconds; runs as user 10001; health check on `/_stcore/health` |
| `.dockerignore` | Keeps `.env`, caches, `.git` and build output out of the image |
| `compose.yaml` | One-command local run; reads `.env` if it exists |
| `deploy/k8s/` | Namespace, Deployment (2 replicas, startup/readiness/liveness probes, resource limits, locked-down security context) and Service (session affinity for Streamlit's websocket), tied together with Kustomize |
| `.github/workflows/docker.yml` | On every push: build, smoke-test with Docker, spin up a kind cluster, deploy, check health through the Service, render every page inside a pod. On `master` it also pushes `ghcr.io/dnaltroc/finance-portfolio:latest` and a commit-SHA tag |

Without `SEC_IDENTITY` the app runs offline on the bundled fixtures and labels them "Sample data (synthetic)".

## Run it on your machine

Install **Docker Desktop** (free for personal use). It sets up WSL 2 on Windows the first time.

### 1. Docker

```bash
docker compose up --build        # first build takes a few minutes
```

Open http://localhost:8501. `Ctrl+C` stops it. Or skip the build and pull the image CI published:

```bash
docker run --rm -p 8501:8501 ghcr.io/dnaltroc/finance-portfolio:latest
```

### 2. Kubernetes

Docker Desktop > **Settings > Kubernetes > Enable Kubernetes** > Apply. `kubectl` comes with it.

```bash
kubectl apply -k deploy/k8s
kubectl -n finance-portfolio get pods -w          # wait for 2/2 Running, Ctrl+C
kubectl -n finance-portfolio port-forward svc/finance-portfolio 8080:80
```

Open http://localhost:8080.

Live SEC data (optional; the key never goes in git or the image):

```bash
kubectl -n finance-portfolio create secret generic finance-portfolio-secrets --from-env-file=.env
kubectl -n finance-portfolio rollout restart deployment/finance-portfolio
```

### 3. Things to try (the interview questions)

| Try | Command | What you will see |
| --- | --- | --- |
| Self-healing | `kubectl -n finance-portfolio delete pod <name>` | A new pod replaces it within seconds |
| Scaling | `kubectl -n finance-portfolio scale deploy/finance-portfolio --replicas=4` | Four pods; scale back to 2 |
| Rolling update | `kubectl -n finance-portfolio rollout restart deploy/finance-portfolio` then `rollout status` | New pods come up before old ones stop (`maxUnavailable: 0`) |
| Probes | `kubectl -n finance-portfolio describe pod <name>` | Startup, readiness and liveness checks on `/_stcore/health` |
| Logs | `kubectl -n finance-portfolio logs deploy/finance-portfolio` | Streamlit's startup output |
| Clean up | `kubectl delete -k deploy/k8s` | Everything removed |

## Design choices to be ready to defend

- **Non-root, read-only root filesystem, all capabilities dropped.** The only writable paths are two `emptyDir` volumes: `/work` for the SQLite cache and `/tmp` for Excel downloads.
- **Startup probe separate from liveness.** The first page load imports pandas and scipy. A startup probe gives it up to two minutes, so liveness never kills a pod that is still starting.
- **Session affinity.** Streamlit holds each browser session over a websocket on one pod, so the Service pins a client to a pod.
- **Image tags.** `latest` for convenience, plus the commit SHA so any deploy can be traced to a commit and rolled back.
- **What it does not have yet.** No Ingress or TLS and no autoscaler. Those are the next steps for a real cluster, and a managed cluster (EKS, GKE, AKS) costs money, so this stays local and in CI.

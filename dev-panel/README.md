# Dev panel (web UI)

Browser-based control panel for the bobr CUDA framework. Runs in WSL or on the GPU host and calls `./bobr` — the same commands agents use from the CLI.

## Quick start

From the repo root:

```bash
./bobr ui
# → http://localhost:1420
```

Or from this directory:

```bash
npm run panel:dev
```

Starts the Express API (port 1421) and Vite dev server (port 1420) together.

## Pages

| Route | Page | Description |
|-------|------|-------------|
| `/` | Home | Overview, doctor status badges, project info |
| `/setup` | Setup | Install, configure, doctor |
| `/build` | Build | Compile C++/CUDA targets |
| `/tests` | Tests | Smoke/full tests, legacy C++ targets, debug scripts |
| `/docs` | Pipeline docs | Build flow and clone checklist |

The output log panel persists across all routes.

## Stack

- **React** + **Vite** SPA
- **React Router** for client-side navigation
- **shadcn/ui** components with a light theme
- **Express** API in `server/` — streams `./bobr` output
- Action definitions in `manifest.json`

## Scripts

| Script | Description |
|--------|-------------|
| `npm run panel:dev` | API + UI dev servers |
| `npm run dev` | Vite only (needs API running separately) |
| `npm run server` | Express API only |
| `npm run build` | Production build to `dist/` |
| `npm run preview` | Preview production build |

## Architecture

```
Browser (localhost:1420)
  ↓ /api/* proxied to Express (localhost:1421)
  ↓ reads manifest.json, runs ./bobr <args>
```

Agents and automation should continue using `./bobr` directly — the panel is a convenience UI, not a required dependency.

# FAME

Unified repository for the **Floor Analysis Mapping Engine (FAME)** prototype. Both the Python
backend that generates the first-pass graphics and the Vite/React interface for drawing borders
and measurement points live here so they can be deployed together.

## Project layout

```
backend/
  FAME.py             # legacy full automation script (requires Google integrations)
  FAME_UI.py          # lightweight Flask service for initial graphics
  requirements.txt    # python dependencies
frontend/
  src/                # React app that captures payloads and displays graphics
  public/data/        # JSON workbook extracts loaded by the UI
  .env.example        # place to configure `VITE_FAME_API_ENDPOINT`
```

> The old `fame-ui/` folder that existed locally is ignored via `.gitignore`. Remove it after
> finishing any local workspaces that still reference it.

## Local development

### Backend (Flask)

```bash
cd backend
python -m venv .venv
.\.venv\Scripts\activate      # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python FAME_UI.py
```

This starts a Flask server on `http://localhost:5000`. The `/api/fame/run` endpoint accepts the
payload produced by the UI and responds with the heatmap, contour, and profile images encoded as
data URIs.

### Frontend (Vite)

```bash
cd frontend
npm install
npm run dev
```

The dev server runs on `http://localhost:5173`. Vite proxies requests that begin with `/api`
to the Flask service, so the button labelled **Run Initial Graphics** works without additional
configuration. To target a remote service (for example a Cloud Run deployment) create
`.env.local` based on `.env.example` and set `VITE_FAME_API_ENDPOINT` to the absolute URL.

```bash
VITE_FAME_API_ENDPOINT="https://your-service-xyz.a.run.app/api/fame/run"
```

## Deployment outline

1. Containerize the backend using `backend/requirements.txt` and deploy to Google Cloud Run.
2. Provide the public URL from Cloud Run via `VITE_FAME_API_ENDPOINT` for the frontend build
   (either through environment variables or updating the `.env` file before running `npm run build`).
3. Host the compiled frontend (`npm run build`) on Firebase Hosting, Cloud Run static build,
   or any static site service.

## Contributing workflow

1. Fork or clone `https://github.com/patrick3395/FAME.git`.
2. Create topic branches for changes (`git checkout -b feature/add-profile-view`).
3. Keep `backend/requirements.txt` and `frontend/package.json` in sync when new dependencies are
   added.
4. Run `npm run build` and the relevant Python smoke tests before opening a pull request.

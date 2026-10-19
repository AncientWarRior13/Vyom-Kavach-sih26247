# Deployment Guide: Vercel & Render

Vyom Kavach is designed for high performance and low latency with a decoupled, modern architecture:

| Component | Recommended Host | Role |
| :--- | :--- | :--- |
| **Frontend Applications** | **Vercel** | Edge CDN distribution for Landing Page (Three.js 3D), Tactical Trainer, AAR Dashboard, Instructor Console, Swarm Simulator & VR views |
| **Backend & ML Engine** | **Render** | Persistent Python runtime hosting FastAPI, ONNX Runtime inference, scenario generators, scoring engine & SQLite audit database |

---

## Part 1: Deploy Backend on Render

### Method A: Render Blueprint (Recommended)

1. Push your repository to GitHub.
2. Log into [Render Dashboard](https://dashboard.render.com/).
3. Click **New +** -> **Blueprint**.
4. Connect your GitHub repository (still named `ps247-dronethreattrainer`/`DroneThreatTrainer` on GitHub — only the product's display name is Vyom Kavach now).
5. Render detects [render.yaml](file:///c:/Users/swast/ps247-drone/render.yaml) automatically.
6. Click **Apply**.
   - Render provisions the Web Service `vyomkavach-api`.
   - Environment variables (`VYOMKAVACH_ALLOW_REMOTE=1`, `PYTHON_VERSION=3.12.8`, `PORT=10000`) are auto-configured.
7. Once deployed, note your service URL:
   `https://vyomkavach-api-sw9g.onrender.com`

### Method B: Docker Web Service

1. On Render, click **New +** -> **Web Service**.
2. Connect your repository.
3. Select **Docker** as the Runtime.
4. Render will automatically build the multi-stage [Dockerfile](file:///c:/Users/swast/ps247-drone/Dockerfile).
5. In **Environment Variables**, ensure:
   - `VYOMKAVACH_ALLOW_REMOTE` = `1`
6. Click **Create Web Service**.

---

## Part 2: Deploy Frontend on Vercel

### Step-by-Step Setup

1. Log into your [Vercel Dashboard](https://vercel.com/dashboard).
2. Click **Add New...** -> **Project**.
3. Import your GitHub repository (still named `ps247-dronethreattrainer`/`DroneThreatTrainer` on GitHub).
4. In the Project Configuration:
   - **Framework Preset**: *Other*
   - **Root Directory**: `./` (leave default repository root)
   - **Build Command**: `npm run build`
   - **Output Directory**: `dist`
5. [vercel.json](file:///c:/Users/swast/ps247-drone/vercel.json) rewrites are already configured to proxy `/api/*`:
   ```json
   "rewrites": [
     {
       "source": "/api/:path*",
       "destination": "https://vyomkavach-api-sw9g.onrender.com/api/:path*"
     }
   ]
   ```
6. Click **Deploy**.

---

## Part 3: Architecture & Security Verification

- **CORS Handling**: `server.py` includes FastAPI `CORSMiddleware` configured to allow cross-origin requests from Vercel deployments.
- **Remote Host Protection**: `VYOMKAVACH_ALLOW_REMOTE=1` instructs the security middleware to allow incoming traffic from cloud routers and edge proxies.
- **Unified Routing**: 
  - `/` -> Landing page & 3D drone trajectory simulation
  - `/trainer` -> Tactical Trainer PPI & 3D urban canopy
  - `/aar` -> Mission debrief and performance analytics
  - `/instructor` -> Live threat injector and jammer console
  - `/team` -> Role-based operational terminal
  - `/swarm` -> Boids algorithm drone swarm simulation
  - `/api/...` -> Reverse-proxied to your live Render backend

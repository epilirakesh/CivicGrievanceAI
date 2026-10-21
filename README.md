# 🏛️ ASTRA — Civic Grievance Intelligence System
### Automated System for Triage, Routing & Analytics for Municipal Grievance Operations

ASTRA is an enterprise-grade civic grievance intelligence platform that unifies citizen complaint intake with a municipal dispatch operations center. It provides multimodal intake (NLP classification, YOLO11n defect vision, IndicConformer Hindi/Marathi voice AI, and GPS Geolocation) coupled with a real-time municipal dashboard for case triage, SLA compliance, and live mapping.

---

## 🌟 Key Capabilities

### 1. 🏢 Administrator Operations Center
- **Needs Attention Bar**: Proactive triage cards for Overdue cases, Unassigned High-Priority issues, and complaints approaching municipal SLA targets.
- **5 Aligned Executive KPIs**:
  1. **Open Issues**: Active complaints currently in the pipeline.
  2. **Overdue**: Critical & High cases that breached resolution targets.
  3. **Unassigned**: Cases awaiting field squad dispatch.
  4. **Average Response Time**: End-to-end triage dispatch latency.
  5. **Average Resolution Time**: Derived strictly from resolved cases.
- **Two-Column Operations Grid (55% / 45%)**:
  - **High-Priority Queue**: Real-time list of critical defects with SLA remaining timers, severity badges, and squad assignment.
  - **Live PyDeck Map**: Real coordinates visualization with status and priority color layers.
- **Recent Reports Table**: Full-width case triage interface with real-count status filter chips (`All`, `Open`, `In Progress`, `Resolved`), search bar, and status/assignment dispatch drawer.

### 2. 👥 Guided Citizen Intake Portal
- **Step 1 — Complaint Details**: Description input with real-time AI category suggestions and narrow-margin review warnings.
- **Step 2 — Multimodal Evidence**:
  - **Photo Vision**: YOLO11n detector (`models/best.pt`) identifying potholes, dangling wires, water leaks, and road defects with annotated bounding boxes.
  - **Voice AI**: AI4Bharat IndicConformer ASR transcribing spoken Hindi (`hi`) and Marathi (`mr`) audio into text and routing to municipal departments. Supports **WhatsApp Audio (`.opus`)**, mobile memos (`.m4a`, `.caf`, `.aac`), and **live in-browser microphone recording**.
- **Step 3 — Geolocation Engine**:
  - One-click **"Use My Current Location"** capturing HTML5 browser GPS coordinates (latitude, longitude, accuracy) with a direct link to Google Maps and manual address/ward fallback.
- **Step 4 — Review & Persistent Submission**:
  - Generates official municipal tracking IDs (`AST-YYYYMMDD-XXXX`), routes to the responsible department, and persists directly in SQLite (`civic_complaints.db`).
- **Track My Complaint**: Dedicated citizen portal for real-time progress tracking, assigned field squads, and dispatcher notes.

---

## 🏗️ Technology Stack

| Layer | Technologies |
|---|---|
| **Frontend & UI** | Streamlit 1.66+, Custom Light Civic CSS Design System, PyDeck |
| **Object Detection** | YOLO11n (Ultralytics) trained for civic defects |
| **Speech-to-Text** | AI4Bharat IndicConformer 600M (Hindi & Marathi CTC decoding) |
| **NLP Classification** | ASTRA 5-Class Linear SVM / Baseline Classifier |
| **Audio Processing** | FFmpeg, SoundFile, Torchaudio |
| **Database** | SQLite3 (`civic_complaints.db`) with schema migration |
| **Deployment** | Docker, Streamlit Cloud, Render, Railway, Linux PaaS |

---

## 🚀 Quickstart & Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/epilirakesh/CivicGrievanceAI.git
cd CivicGrievanceAI
```

### 2. Create virtual environment & install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Ensure FFmpeg is available
- **macOS**: `brew install ffmpeg`
- **Ubuntu/Debian**: `sudo apt-get install ffmpeg`
- **Windows**: `choco install ffmpeg`

### 4. Run the application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ☁️ Cloud Deployment Options

### Option A: Streamlit Community Cloud (Recommended for 1-Click)
1. Fork or push this repository to your GitHub account (`epilirakesh/CivicGrievanceAI`).
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with GitHub.
3. Select this repository: `epilirakesh/CivicGrievanceAI`, branch: `main`, and main file: `app.py`.
4. (Optional) In **Advanced Settings**, add secrets:
   ```toml
   HF_TOKEN = "your_huggingface_token"
   ```
5. Click **Deploy**. Streamlit Cloud will automatically use `packages.txt` to install FFmpeg and `requirements.txt` for Python dependencies.

### Option B: Docker Container
A production `Dockerfile` is included.

```bash
# Build the Docker image
docker build -t civic-grievance-ai .

# Run the container
docker run -p 8501:8501 civic-grievance-ai
```

### Option C: Render / Railway / Heroku
The included `Procfile` enables zero-config deployment:
```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
```

---

## 🧪 Automated Test Suites

Run the test suite to verify vision, speech, database, and triage features:

```bash
# Verify fixes & multimodal vision integrity
python tests/verify_fixes.py

# Verify SQLite database schema & triage workflows
python tests/test_database.py

# Verify multimodal intake service
python tests/test_grievance_service.py
```

---

## 📄 License
This project is licensed under the Apache 2.0 / MIT License.

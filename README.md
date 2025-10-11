AI Business Insight Dashboard
================================
Built for Caprae Capital - AI-Readiness Full Stack Challenge
An AI-powered full-stack dashboard that ingests business CSVs and produces:
- GPT-generated concise dataset summaries,
- automatic numeric statistics and date detection,
- quick time-series forecasts (linear regression),
- an interactive chat assistant that answers dataset-specific questions.
This demo showcases a complete end-to-end pipeline: FastAPI backend (data parsing, AI
endpoints, forecasting), React frontend (file upload, charts, chat), and OpenAI integration.
------------------------------------------------------------
Demo Video: https://www.dropbox.com/scl/fi/u0udp6oobmjbjlnblxru9/ref_video.mp4?rlkey=6mwidpi5ofnrayu08lydkwi7v&e=1&st=rwtebfi6&dl=0
------------------------------------------------------------
Features
------------------------------------------------------------
- CSV upload & robust parsing (pandas, fallback parser)
- AI summary endpoint: /ai/summary (returns session_id)
- Chat assistant with session memory: /ai/chat (multi-turn for same dataset)
- Forecast endpoint: /forecast (simple, fast linear regression)
- Mock mode available for offline demos (use_mock=true)
- CORS enabled for local dev; ready to deploy
------------------------------------------------------------
Tech Stack
------------------------------------------------------------
Backend: Python, FastAPI, pandas, scikit-learn (LinearRegression), openai (optional)
Frontend: React (create-react-app), axios, Chart.js (react-chartjs-2)
Dev tooling: uvicorn, npm, python-dotenv
------------------------------------------------------------
Quickstart (Windows / PowerShell)
------------------------------------------------------------
1. Backend
 cd backend
 .\venv\Scripts\Activate.ps1
 pip install -r requirements.txt
 uvicorn main:app --reload --port 8000
2. Frontend
 cd frontend
 npm install
 npm start
3. Demo flow
 - Upload sample_data/sales_data.csv in the UI
 - Watch AI summary, numeric stats, and forecast chart appear
 - Use chat panel (uses session_id) for follow-ups
------------------------------------------------------------
API Quick Reference
------------------------------------------------------------
GET /health
POST /ai/summary - upload CSV, returns session_id, summary, stats
POST /ai/chat - form: message + session_id or file
POST /forecast - form: file + date_column, target, horizon
------------------------------------------------------------
Contact
------------------------------------------------------------
Baibhab Ghosh
Email: baibhabghosh2003@gmail.com
GitHub: https://github.com/MonGeR-B/AI-BusinessInsight-Dashboard


from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import io
import numpy as np
from typing import Any, Dict, Optional
import os
from sklearn.linear_model import LinearRegression
import json
import uuid
import openai

app = FastAPI(title="Investor-Connect Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


DATA_STORE: Dict[str, Dict] = {}

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)
client = None
if OPENAI_API_KEY:
    client = openai.OpenAI(api_key=OPENAI_API_KEY)


def try_parse_dates(df: pd.DataFrame) -> Dict[str, str]:
    detected = {}
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            non_null = parsed.notnull().sum()
            if non_null >= max(1, int(0.5 * len(parsed))):
                detected[col] = str(parsed.dtype)
        except Exception:
            continue
    return detected

def numeric_stats(df: pd.DataFrame) -> Dict[str, Any]:
    stats = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        s = df[col].dropna()
        stats[col] = {
            "count": int(s.count()),
            "mean": float(s.mean()) if s.count() else None,
            "std": float(s.std()) if s.count() else None,
            "min": float(s.min()) if s.count() else None,
            "25%": float(s.quantile(0.25)) if s.count() else None,
            "50%": float(s.median()) if s.count() else None,
            "75%": float(s.quantile(0.75)) if s.count() else None,
            "max": float(s.max()) if s.count() else None,
        }
    return stats

def df_from_upload(contents: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(contents))
    except Exception:
        return pd.read_csv(io.BytesIO(contents), engine="python", encoding="utf-8-sig")

def build_basic_meta(df: pd.DataFrame, filename: str) -> Dict[str, Any]:
    return {
        "filename": filename,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }

def create_session_from_df(df: pd.DataFrame, filename: str) -> str:
    session_id = str(uuid.uuid4())
    meta = build_basic_meta(df, filename)
    head = df.head(5).replace({np.nan: None}).to_dict(orient="records")
    stats = numeric_stats(df)
    dates = try_parse_dates(df)
    DATA_STORE[session_id] = {
        "df": df,
        "meta": meta,
        "head": head,
        "numeric_stats": stats,
        "dates": dates,
        "messages": [],
    }
    return session_id

def build_chat_context(session: Dict, user_message: str) -> str:
    meta = session.get("meta", {})
    stats = session.get("numeric_stats", {})
    head = session.get("head", [])
    dates = session.get("dates", {})

    short_stats = {k: {"mean": v.get("mean"), "min": v.get("min"), "max": v.get("max")} for k, v in stats.items()}
    lines = [
        "You are a concise data analyst assistant. Use the provided dataset context to answer the user's question.",
        "",
        "DATA META:",
        json.dumps(meta, indent=2),
        "",
        "NUMERIC_STATS (sample):",
        json.dumps(short_stats, indent=2),
        "",
        "HEAD (first rows):",
        json.dumps(head, indent=2),
        "",
        "DETECTED_DATE_COLUMNS:",
        json.dumps(dates, indent=2),
        "",
        "CONVERSATION:",
    ]
    for m in session.get("messages", [])[-10:]:
        lines.append(f"{m.get('role')}: {m.get('text')}")
    lines.extend([
        f"User: {user_message}",
        "",
        "Answer in 2-5 sentences aimed at a product manager. Be direct. No code blocks. If the user asks for predictions, recommend using /forecast and give short instructions.",
    ])
    return "\n".join(lines)

def mock_chat_response(session: Dict, user_message: str) -> str:
    stats = session.get("numeric_stats", {})
    head = session.get("head", [])
    dates = session.get("dates", {})
    parts = []
    if stats:
        variances = {k: (v["std"] or 0) for k, v in stats.items()}
        top = max(variances.keys(), key=lambda k: variances[k]) if variances else None
        if top:
            s = stats[top]
            parts.append(f"{top} has mean {round(s['mean'],2)} and range {round(s['min'],2)}–{round(s['max'],2)}.")
    if dates:
        parts.append(f"Detected date columns: {', '.join(dates.keys())}. Use these for time-series analysis.")
    msg_lower = user_message.lower()
    if "risk" in msg_lower or "issue" in msg_lower:
        parts.append("Primary risks look like volatility in the top numeric columns; consider deeper outlier analysis and more data points.")
    elif "forecast" in msg_lower or "predict" in msg_lower:
        parts.append("Use the /forecast endpoint with the detected date column and target numeric column; horizon can be adjusted.")
    else:
        parts.append("If you want a prediction or visualization, ask for a forecast or request charts.")
    return " ".join(parts)

async def call_openai_chat(prompt: str, model: str = "gpt-4o-mini") -> str:
    if not client:
        raise RuntimeError("OPENAI_API_KEY not set")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.2,
        )
        if resp.choices and len(resp.choices) > 0:
            content = resp.choices[0].message.content
            return content.strip() if content else ""
        return str(resp)
    except Exception as e:
        raise

@app.get("/health")
async def health():
    return {"status": "ok", "service": "investor-connect-backend", "version": "day4"}

@app.post("/ai/summary")
async def ai_summary(file: UploadFile = File(...), use_mock: Optional[bool] = Query(False, description="If true, skip calling OpenAI and return a deterministic mock summary")):
    filename = file.filename
    if not filename or not filename.lower().endswith((".csv", ".txt")):
        raise HTTPException(status_code=400, detail="Only CSV or TXT files are accepted.")
    contents = await file.read()
    try:
        df = df_from_upload(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV. Error: {str(e)}")
    session_id = create_session_from_df(df, filename)
    session = DATA_STORE[session_id]
    meta = session["meta"]
    head = session["head"]
    stats = session["numeric_stats"]
    dates = session["dates"]

    prompt = build_chat_context(session, "Please provide a concise summary of the dataset.")
    if use_mock or not OPENAI_API_KEY:
        summary = mock_chat_response(session, "Please provide a concise summary of the dataset.")
        mode = "mock"
    else:
        try:
            summary = await call_openai_chat(prompt)
            mode = "openai"
        except Exception as e:
            summary = mock_chat_response(session, "Please provide a concise summary of the dataset.") + f"\n\n(Note: OpenAI call failed: {str(e)})"
            mode = "fallback"

    return {
        "session_id": session_id,
        "meta": meta,
        "head": head,
        "numeric_stats": stats,
        "detected_date_columns": dates,
        "summary": summary,
        "mode": mode,
    }

@app.post("/ai/chat")
async def ai_chat(
    message: str = Form(...),
    session_id: Optional[str] = Query(None, description="Existing session_id to use (preferred)."),
    file: Optional[UploadFile] = File(None),
    use_mock: Optional[bool] = Query(False, description="If true, skip OpenAI and return mock response")
):

    session = None
    used_new_session = False

    if session_id:
        session = DATA_STORE.get(session_id)
        if not session:
            raise HTTPException(status_code=400, detail=f"session_id '{session_id}' not found.")
    elif file:

        filename = file.filename
        if not filename or not filename.lower().endswith((".csv", ".txt")):
            raise HTTPException(status_code=400, detail="Only CSV or TXT files are accepted.")
        contents = await file.read()
        try:
            df = df_from_upload(contents)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse CSV. Error: {str(e)}")
        new_session_id = create_session_from_df(df, filename or "unknown.csv")
        session = DATA_STORE[new_session_id]
        session_id = new_session_id
        used_new_session = True
    else:
        raise HTTPException(status_code=400, detail="Provide either session_id or file (CSV) with the request.")


    session["messages"].append({"role": "user", "text": message})

    session["messages"] = session["messages"][-10:]


    prompt = build_chat_context(session, message)

    if use_mock or not OPENAI_API_KEY:
        reply = mock_chat_response(session, message)
        mode = "mock"
    else:
        try:
            reply = await call_openai_chat(prompt)
            mode = "openai"
        except Exception as e:
            reply = mock_chat_response(session, message) + f"\n\n(Note: OpenAI call failed: {str(e)})"
            mode = "fallback"


    session["messages"].append({"role": "assistant", "text": reply})
    session["messages"] = session["messages"][-10:]

    return {
        "session_id": session_id,
        "reply": reply,
        "mode": mode,
        "messages_in_session": len(session["messages"]),
        "used_new_session": used_new_session,
    }

@app.post("/forecast")
async def forecast(file: UploadFile = File(...), target: Optional[str] = Query(None, description="Numeric column name to forecast; if omitted, top numeric column is chosen."), date_column: Optional[str] = Query(None, description="Name of date column to use for time index; if omitted the detected date column will be used."), horizon: Optional[int] = Query(3, description="Number of future points to predict (integer)")):
    horizon = horizon or 3
    filename = file.filename
    if not filename or not filename.lower().endswith((".csv", ".txt")):
        raise HTTPException(status_code=400, detail="Only CSV or TXT files are accepted.")
    contents = await file.read()
    try:
        df = df_from_upload(contents)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV. Error: {str(e)}")

    dates = try_parse_dates(df)
    date_col = date_column
    if not date_col:
        date_col = next(iter(dates.keys()), None)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise HTTPException(status_code=400, detail="No numeric columns available for forecasting.")
    tgt = target if target in numeric_cols else (numeric_cols[0] if target is None else None)
    if tgt is None:
        raise HTTPException(status_code=400, detail=f"Requested target '{target}' not found in numeric columns: {numeric_cols}")

    df_clean = df[[tgt]].copy()
    if date_col:
        try:
            df_clean[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df_clean = df_clean.dropna(subset=[date_col, tgt])
            df_clean = df_clean.sort_values(by=date_col)
            X = np.array(df_clean[date_col].map(pd.Timestamp.toordinal)).reshape(-1, 1)
            y = np.array(df_clean[tgt]).reshape(-1, 1)
            last_date = df_clean[date_col].max()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to prepare time index using {date_col}: {str(e)}")
    else:
        df_clean = df_clean.dropna(subset=[tgt])
        X = np.arange(len(df_clean)).reshape(-1, 1)
        y = np.array(df_clean[tgt]).reshape(-1, 1)
        last_date = None

    if len(X) < 3:
        raise HTTPException(status_code=400, detail="Not enough data points for reliable forecast (need at least 3).")

    model = LinearRegression()
    model.fit(X, y)

    if date_col and last_date is not None:
        if len(X) >= 2:
            steps = np.diff(sorted(df_clean[date_col].map(pd.Timestamp.toordinal).values))
            avg_step = int(round(np.mean(steps))) if len(steps) > 0 else 1
        else:
            avg_step = 1
        start = pd.Timestamp(last_date).toordinal()
        future_ordinals = np.array([start + avg_step * i for i in range(1, horizon + 1)]).reshape(-1, 1)
        preds = model.predict(future_ordinals)
        future_dates = [pd.Timestamp.fromordinal(int(int(x))) for x in future_ordinals.flatten()]
        predictions = [{"date": str(d.date()), "prediction": float(p[0])} for d, p in zip(future_dates, preds)]
    else:
        last_idx = int(X.max())
        future_idx = np.array([last_idx + i for i in range(1, horizon + 1)]).reshape(-1, 1)
        preds = model.predict(future_idx)
        predictions = [{"index": int(idx[0]), "prediction": float(p[0])} for idx, p in zip(future_idx, preds)]

    return {
        "meta": build_basic_meta(df, filename or "unknown.csv"),
        "target": tgt,
        "date_column_used": date_col,
        "horizon": horizon,
        "predictions": predictions,
        "model": "LinearRegression (sklearn)",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

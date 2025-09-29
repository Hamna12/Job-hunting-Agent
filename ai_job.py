import argparse, os, re, sqlite3, sys, json, time
from datetime import datetime, timezone
from typing import List, Dict, Any
import requests, pandas as pd, feedparser
from pydantic import BaseModel, Field

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
OUT_DIR = "out"
DB_PATH = os.path.join(OUT_DIR, "jobs.sqlite")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "drafts"), exist_ok=True)

# ----------------------------- Data Models -----------------------------
class Job(BaseModel):
    id: str
    source: str
    title: str
    company: str
    url: str
    location: str = "Remote"
    posted_at: str = ""
    tags: List[str] = Field(default_factory=list)
    summary: str = ""

class Analysis(BaseModel):
    relevance_score: int
    key_skills: List[str]
    contact_priority: str   
    outreach_subject: str
    outreach_email: str

# ----------------------------- Storage -----------------------------
def db_init():
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS jobs(
        id TEXT PRIMARY KEY,
        source TEXT,
        title TEXT,
        company TEXT,
        url TEXT,
        location TEXT,
        posted_at TEXT,
        tags TEXT,
        summary TEXT,
        added_at TEXT
    )""")
    cur.execute("""
    CREATE TABLE IF NOT EXISTS applications(
        job_id TEXT PRIMARY KEY,
        relevance INTEGER,
        priority TEXT,
        subject TEXT,
        email TEXT,
        created_at TEXT
    )""")
    con.commit()
    con.close()

def job_seen(job_id: str) -> bool:
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("SELECT 1 FROM jobs WHERE id = ?", (job_id,))
    row = cur.fetchone()
    con.close()
    return row is not None

def save_job(job: Job):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        INSERT OR IGNORE INTO jobs(id, source, title, company, url, location, posted_at, tags, summary, added_at)
        VALUES(?,?,?,?,?,?,?,?,?,?)
    """, (job.id, job.source, job.title, job.company, job.url, job.location, job.posted_at,
          ",".join(job.tags), job.summary, datetime.utcnow().isoformat()))
    con.commit()
    con.close()

def save_application(job_id: str, a: Analysis):
    con = sqlite3.connect(DB_PATH)
    cur = con.cursor()
    cur.execute("""
        INSERT OR REPLACE INTO applications(job_id, relevance, priority, subject, email, created_at)
        VALUES(?,?,?,?,?,?)
    """, (job_id, a.relevance_score, a.contact_priority, a.outreach_subject, a.outreach_email,
          datetime.utcnow().isoformat()))
    con.commit()
    con.close()

# ----------------------------- Sources -----------------------------
def fetch_remotive(keywords: List[str], limit: int) -> List[Job]:
    """Official API: https://remotive.com/api/remote-jobs (24h delayed, rate-limited)"""
    base = "https://remotive.com/api/remote-jobs"
    out: List[Job] = []
    seen = set()
    for kw in keywords:
        params = {"search": kw, "limit": limit}
        try:
            r = requests.get(base, params=params, timeout=20)
            r.raise_for_status()
            data = r.json()
            for j in data.get("jobs", []):
                jid = f"remotive:{j['id']}"
                if jid in seen: continue
                seen.add(jid)
                out.append(Job(
                    id=jid,
                    source="Remotive",
                    title=j.get("title","").strip(),
                    company=j.get("company_name","").strip(),
                    url=j.get("url",""),
                    location=j.get("candidate_required_location","Remote"),
                    posted_at=j.get("publication_date",""),
                    tags=j.get("tags",[]) or [],
                    summary=re.sub("<.*?>"," ", j.get("description",""))[:800].strip()
                ))
        except Exception as e:
            print(f"[remotive] error for '{kw}': {e}")
    return out[:limit]

def fetch_remoteok_try(keywords: List[str], limit: int) -> List[Job]:
    """
    Optional secondary source. RemoteOK offers JSON feeds; endpoints vary.
    We try a couple and fail silently if blocked.
    """
    endpoints = [
        "https://remoteok.com/remote-jobs.json",  
        "https://remoteok.io/api"                 
    ]
    out: List[Job] = []
    headers = {"User-Agent": "Mozilla/5.0 (JobAgent/1.0)"}
    for url in endpoints:
        try:
            r = requests.get(url, headers=headers, timeout=20)
            if r.status_code != 200: 
                continue
            data = r.json()
            # Some feeds have a banner object at index 0
            jobs = data if isinstance(data, list) else data.get("jobs", [])
            for j in jobs:
                title = j.get("position") or j.get("title") or ""
                if not title: continue
                # Keyword filter
                hay = " ".join([title, j.get("description",""), " ".join(j.get("tags",[]) or [])]).lower()
                if not any(kw.lower() in hay for kw in keywords): 
                    continue
                jid = f"remoteok:{j.get('id') or j.get('slug') or j.get('url')}"
                out.append(Job(
                    id=jid,
                    source="RemoteOK",
                    title=title.strip(),
                    company=(j.get("company") or j.get("company_name") or "").strip(),
                    url=j.get("url") or j.get("apply_url") or "",
                    location=(j.get("location") or "Remote"),
                    posted_at=(j.get("date") or j.get("created_at") or ""),
                    tags=j.get("tags",[]) or [],
                    summary=re.sub("<.*?>"," ", j.get("description",""))[:800].strip()
                ))
            if out: break  # one endpoint worked, good enough
        except Exception:
            continue
    # Dedup by URL
    uniq = {}
    for j in out:
        uniq[j.url] = j
    return list(uniq.values())[:limit]

# ----------------------------- Ollama -----------------------------
def ollama_generate_json(system: str, user: str, temperature: float = 0.2, max_tokens: int = 800) -> Dict[str, Any]:
    """
    Call Ollama and force JSON output using 'format': 'json'.
    """
    payload = {
        "model": MODEL,
        "prompt": f"System: {system}\n\nUser: {user}",
        "stream": False,
        "format": "json",
        "options": {"temperature": temperature, "num_ctx": 8192}
    }
    try:
        res = requests.post(OLLAMA_URL, json=payload, timeout=90)
        res.raise_for_status()
        txt = res.json().get("response","").strip()
        return json.loads(txt) if txt else {}
    except Exception as e:
        print(f"[ollama] generation failed: {e}")
        return {}

def analyze_job(job: Job) -> Analysis:
    system = "You are a sharp startup BD analyst for a 2-person GenAI agency. Return strict JSON."
    user = f"""
Evaluate this opportunity for a small agency offering Agents, RAG, LLM fine-tuning, CV/NLP.

JOB:
Title: {job.title}
Company: {job.company}
Location: {job.location}
Tags: {', '.join(job.tags)}
Summary: {job.summary[:600]}

Return JSON with exactly these keys:
- relevance_score (int 1-10: 40% AI/GenAI relevance, 30% startup/contract friendliness, 30% skills fit)
- key_skills (array of strings)
- contact_priority (one of: High, Medium, Low)
- outreach_subject (short, specific)
- outreach_email (≤150 words, friendly-professional, references the role, suggests a brief call)
"""
    out = ollama_generate_json(system, user)
    # Fallback if LLM drifts
    if not out or "relevance_score" not in out:
        # simple heuristic
        text = (job.title + " " + job.summary + " " + " ".join(job.tags)).lower()
        score = 5 + sum(1 for k in ["ai","llm","agent","rag","nlp","vision","ml"] if k in text)
        score = min(score, 9)
        out = {
            "relevance_score": score,
            "key_skills": ["Python","LLMs","RAG"] if score >=7 else ["Python","ML"],
            "contact_priority": "High" if score >=8 else ("Medium" if score>=6 else "Low"),
            "outreach_subject": f"{job.title} — quick intro re: GenAI help",
            "outreach_email": (
                f"Hi {job.company} team,\n\n"
                f"We’re a 2-person GenAI studio (agents, RAG, fine-tuning). "
                f"Saw your '{job.title}' role and believe we can deliver fast. "
                f"Happy to share 1–2 relevant builds and propose a scoped pilot.\n\n"
                f"Open to a 15-min chat this week?\n— Hamna & Arslan"
            )
        }
    # sanitize
    out["relevance_score"] = int(out.get("relevance_score", 5))
    out["key_skills"] = out.get("key_skills", [])[:10]
    out["contact_priority"] = str(out.get("contact_priority","Medium")).title()
    out["outreach_subject"] = out.get("outreach_subject","Quick intro re: GenAI help")[:120]
    out["outreach_email"] = out.get("outreach_email","")[:1200]
    return Analysis(**out)

# ----------------------------- Pipeline -----------------------------
def run_once(keywords: List[str], limit: int, silent: bool=False) -> pd.DataFrame:
    db_init()

    # 1) Fetch
    remotive_jobs = fetch_remotive(keywords, limit)
    remoteok_jobs = fetch_remoteok_try(keywords, limit//2)
    jobs = remotive_jobs + remoteok_jobs

    # 2) Dedup & store
    clean: List[Job] = []
    seen_urls = set()
    for j in jobs:
        if not j.url or j.url in seen_urls:
            continue
        seen_urls.add(j.url)
        if not job_seen(j.id):
            save_job(j)
        clean.append(j)

    if not silent:
        print(f"Fetched: {len(jobs)} | After dedup: {len(clean)}")

    # 3) Analyze + Draft
    rows = []
    for j in clean:
        a = analyze_job(j)
        save_application(j.id, a)

        # write draft txt
        fn = os.path.join(OUT_DIR, "drafts", f"{j.id.replace(':','_')}.txt")
        with open(fn, "w", encoding="utf-8") as f:
            f.write(f"Subject: {a.outreach_subject}\nURL: {j.url}\n\n{a.outreach_email}\n")

        rows.append({
            "source": j.source,
            "title": j.title,
            "company": j.company,
            "location": j.location,
            "posted_at": j.posted_at,
            "score": a.relevance_score,
            "priority": a.contact_priority,
            "key_skills": ", ".join(a.key_skills),
            "url": j.url,
            "draft_file": fn
        })

    # 4) Sort & export CSV
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["priority","score"], ascending=[True, False])
        today = datetime.now().strftime("%Y-%m-%d")
        out_csv = os.path.join(OUT_DIR, f"jobs_{today}.csv")
        df.to_csv(out_csv, index=False, encoding="utf-8")
        if not silent:
            print(f"Saved {len(df)} rows -> {out_csv}")
    return df

# ----------------------------- CLI -----------------------------
def main():
    ap = argparse.ArgumentParser(description="AI Job Hunting Agent (Ollama + JSON feeds)")
    ap.add_argument("--keywords", type=str, default="ai,llm,agent,rag,nlp,computer vision,generative")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--silent", action="store_true")
    args = ap.parse_args()

    # quick Ollama ping
    try:
        requests.get("http://localhost:11434/api/tags", timeout=4)
    except Exception:
        print("⚠️  Ollama not reachable at http://localhost:11434. Start it with `ollama serve`.")
        # continue anyway; we'll use heuristic fallback

    keywords = [k.strip() for k in args.keywords.split(",") if k.strip()]
    df = run_once(keywords, args.limit, silent=args.silent)
    if df is None or df.empty:
        print("No jobs found today. Try increasing --limit or adjusting keywords.")
    else:
        print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()

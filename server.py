import os, glob, json, math, shlex, signal, subprocess, threading, time, uuid
import httpx
from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

KEY = os.environ.get("GROQ_API_KEY", "")
BASE = "/tmp/clipper"; SEG = f"{BASE}/seg"; OUT = f"{BASE}/clips"
SEG_LEN = 5; KEEP = 130  # ~10 min buffer
os.makedirs(SEG, exist_ok=True); os.makedirs(OUT, exist_ok=True)
state = {"proc": None, "url": "", "auto": False}
app = FastAPI()

def segs(): return sorted(glob.glob(f"{SEG}/s_*.ts"))

def stop():
    p = state["proc"]
    if p:
        try: os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except Exception: pass
    state["proc"] = None

def start(url):
    stop()
    for f in segs(): os.remove(f)
    cmd = (f"streamlink --stdout {shlex.quote(url)} best | ffmpeg -loglevel error -i pipe:0 "
           f"-c copy -f segment -segment_time {SEG_LEN} -reset_timestamps 1 {SEG}/s_%06d.ts")
    state["proc"] = subprocess.Popen(cmd, shell=True, preexec_fn=os.setsid)
    state["url"] = url

def tail(secs):
    n = math.ceil(secs / SEG_LEN)
    files = segs()[-(n + 1):-1]
    if not files: raise HTTPException(400, "Abhi buffer khali hai, kuch der ruko")
    lst = f"{BASE}/{uuid.uuid4().hex}.txt"
    open(lst, "w").write("".join(f"file '{f}'\n" for f in files))
    return lst

def make_clip(secs):
    lst = tail(secs); name = f"clip_{int(time.time())}.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
                    "-c", "copy", f"{OUT}/{name}"], check=True)
    os.remove(lst); return name

def groq_analyze(secs):
    KEY = state.get("key") or os.environ.get("GROQ_API_KEY", "")
    if not KEY: raise HTTPException(400, "GROQ_API_KEY set nahi hai")
    lst = tail(secs); mp3 = f"{BASE}/a.mp3"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst,
                    "-vn", "-ac", "1", "-ar", "16000", "-b:a", "32k", mp3], check=True)
    os.remove(lst); h = {"Authorization": f"Bearer {KEY}"}
    r = httpx.post("https://api.groq.com/openai/v1/audio/transcriptions", headers=h,
                   files={"file": open(mp3, "rb")}, data={"model": "whisper-large-v3-turbo"}, timeout=120)
    r.raise_for_status(); text = r.json()["text"]
    prompt = ("Ye live stream ka transcript hai. Batao kya ye viral/highlight-worthy moment hai. "
              'Sirf JSON do: {"score": 1-10, "title": "short catchy title"}\n\n' + text[:6000])
    c = httpx.post("https://api.groq.com/openai/v1/chat/completions", headers=h, timeout=60, json={
        "model": "llama-3.3-70b-versatile", "response_format": {"type": "json_object"},
        "messages": [{"role": "user", "content": prompt}]})
    c.raise_for_status(); j = json.loads(c.json()["choices"][0]["message"]["content"])
    return text, int(j.get("score", 0)), str(j.get("title", "clip"))

def worker():
    last = 0
    while True:
        time.sleep(10)
        for f in segs()[:-KEEP]:
            try: os.remove(f)
            except Exception: pass
        if state["auto"] and state["proc"] and time.time() - last > 60:
            last = time.time()
            try:
                _, score, title = groq_analyze(60)
                if score >= 7: make_clip(60)
            except Exception as e: print("auto err", e)
threading.Thread(target=worker, daemon=True).start()

class U(BaseModel): url: str
class S(BaseModel): seconds: int = 30
class A(BaseModel):
    on: bool
    key: str = ""

@app.post("/api/start")
def a_start(b: U): start(b.url); return {"ok": True}
@app.post("/api/stop")
def a_stop(): stop(); return {"ok": True}
@app.post("/api/auto")
def a_auto(b: A):
    if b.key: state["key"] = b.key
    state["auto"] = b.on; return {"auto": b.on}
@app.get("/api/status")
def a_status():
    return {"running": bool(state["proc"] and state["proc"].poll() is None), "url": state["url"],
            "buffer": len(segs()) * SEG_LEN, "auto": state["auto"],
            "clips": sorted(os.listdir(OUT), reverse=True)[:20]}
@app.post("/api/clip")
def a_clip(b: S): return {"file": make_clip(min(b.seconds, 600))}
@app.post("/api/ai")
def a_ai(b: S, x_groq_key: str = Header(None)):
    if x_groq_key: state["key"] = x_groq_key
    text, score, title = groq_analyze(b.seconds)
    return {"file": make_clip(b.seconds), "score": score, "title": title, "transcript": text}
@app.get("/clips/{f}")
def a_file(f: str): return FileResponse(f"{OUT}/{os.path.basename(f)}", media_type="video/mp4")
app.mount("/", StaticFiles(directory="static", html=True), name="static")

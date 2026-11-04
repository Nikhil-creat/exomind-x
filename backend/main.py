"""ExoMind-X API: FastAPI gateway + static site. Demo scoring is a transparent heuristic;
swap `score()` for ONNX inference of the trained ExoNet-V2 (see ml/model.py)."""
import json, math, re, time, os
from collections import defaultdict
from pathlib import Path
from fastapi import FastAPI, Request, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

BASE = Path(__file__).parent
app = FastAPI(title="ExoMind-X Enterprise", version="1.0.0")
M = defaultdict(int)
BUCKETS: dict = {}
HITL: list = []
CAP, RATE = 30.0, 0.5  # token bucket: 30 burst, 1 token / 2s

@app.middleware("http")
async def guard(req: Request, call_next):
    if req.url.path.startswith("/api"):
        ip = req.headers.get("x-forwarded-for", req.client.host if req.client else "x").split(",")[0]
        t = time.time(); tok, last = BUCKETS.get(ip, (CAP, t))
        tok = min(CAP, tok + (t - last) * RATE)
        if tok < 1:
            M["rate_limited_total"] += 1
            return JSONResponse({"detail": "rate limit exceeded"}, 429)
        BUCKETS[ip] = (tok - 1, t); M["api_requests_total"] += 1
    r = await call_next(req)
    r.headers.update({"X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY", "Referrer-Policy": "no-referrer"})
    return r

class TCE(BaseModel):
    kepid: int = 10593626
    period: float = Field(gt=0, le=1000)
    duration: float = Field(gt=0, le=48, description="hours")
    depth_ppm: float = Field(gt=0, le=1e6)
    snr: float = Field(ge=0, le=500)
    secondary_ppm: float = Field(0, ge=0, le=1e6)

sig = lambda z: 1 / (1 + math.exp(-z))

def score(t: TCE):
    duty = t.duration / (t.period * 24)
    sec = t.secondary_ppm / t.depth_ppm
    z = 1.15 * math.log1p(t.snr) - 3.2 - 4.5 * sec - (1.6 if duty > 0.12 else 0) - (1.8 if t.depth_ppm > 40000 else 0)
    base = sig(0.7 * math.log1p(t.snr) - 1.6 - (1.2 if t.depth_ppm > 40000 else 0))
    return sig(z), base, duty, sec

CORPUS = [
 ("Secondary eclipse test", "A secondary dip near phase 0.5 of comparable depth indicates an eclipsing binary, not a planet. Planets show only tiny occultations."),
 ("Odd-even depth test", "Compare alternating transit depths. A significant mismatch suggests a binary at twice the period."),
 ("V-shaped vs U-shaped dips", "Flat-bottomed U-shaped transits favor planets; V-shaped grazing dips with long ingress favor stellar companions."),
 ("Dual-view CNN vetting", "Global (2001 bin) and local (201 bin) folded light curve views feed convolutional branches, as in AstroNet-style exoplanet vetting (Shallue and Vanderburg, 2018)."),
 ("Centroid and contamination", "Centroid shifts during transit reveal background eclipsing binaries blended into the aperture."),
 ("Giant planet false positives", "Depths above several percent imply stellar-radius companions; Jupiter-size planets give roughly one percent depth."),
 ("Detrending with transit masks", "Mask known transits before flattening so the filter does not erase the signal; long windows preserve transit shape."),
]
tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())
def bm25(q, k=3):
    docs = [tok(a + " " + b) for a, b in CORPUS]; avg = sum(map(len, docs)) / len(docs); out = []
    for i, d in enumerate(docs):
        s = 0
        for w in set(tok(q)):
            n = sum(w in x for x in docs); idf = math.log(1 + (len(docs) - n + .5) / (n + .5)); f = d.count(w)
            s += idf * f * 2.2 / (f + 1.2 * (.25 + .75 * len(d) / avg))
        out.append((s, i))
    return [{"title": CORPUS[i][0], "note": CORPUS[i][1], "score": round(s, 3)} for s, i in sorted(out, reverse=True)[:k] if s > 0]

@app.get("/api/v1/health")
def health(): return {"status": "ok", "service": "exomind-x", "version": app.version}

@app.post("/api/v1/predict")
def predict(t: TCE):
    p, b, duty, sec = score(t); M["predictions_total"] += 1
    return {"kepid": t.kepid, "p_planet": round(p, 4), "baseline": round(b, 4), "label": "candidate" if p > .65 else "false_positive" if p < .40 else "ambiguous"}

@app.post("/api/v1/agent/analyze")
def analyze(t: TCE):
    """Ingestion -> Classifier -> Diagnostic -> RAG (LangGraph-style state machine, dependency-free)."""
    p, b, duty, sec = score(t); M["analyses_total"] += 1
    trace = [{"agent": "IngestionAgent", "out": f"validated TCE, duty cycle {duty:.2%}"},
        {"agent": "ClassifierAgent", "out": f"ExoNet P={p:.3f} vs baseline {b:.3f}"}]
    anomaly = .40 <= p <= .65 or abs(p - b) > .25
    flags = []
    if sec > .3: flags.append("secondary eclipse at phase 0.5")
    if duty > .12: flags.append("long duty cycle (V-shape risk)")
    if t.depth_ppm > 40000: flags.append("stellar-scale depth")
    trace.append({"agent": "DiagnosticAgent", "out": "; ".join(flags) or "no eclipse or V-shape flags"})
    rag = bm25(" ".join(flags) + (" CNN vetting" if anomaly else "")) if (flags or anomaly) else []
    trace.append({"agent": "RAGAgent", "out": f"{len(rag)} passages retrieved (BM25)" if rag else "skipped (confident, clean)"})
    hitl = .45 <= p <= .55
    if hitl:
        HITL.insert(0, {"kepid": t.kepid, "p": round(p, 3), "ts": int(time.time())}); del HITL[50:]
    verdict = ("Likely planet candidate" if p > .65 and not flags else "Likely false positive" if p < .40 or sec > .3 else "Ambiguous, routed for expert review")
    return {"kepid": t.kepid, "p_planet": round(p, 4), "baseline": round(b, 4), "is_anomaly": anomaly, "flags": flags,
            "trace": trace, "rag_context": rag, "queued_for_hitl": hitl, "final_verdict_report": verdict}

@app.get("/api/v1/hitl")
def hitl(): return HITL[:10]

class Msg(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    message: str = Field(min_length=3, max_length=2000)

@app.post("/api/v1/contact")
def contact(m: Msg):
    with open(os.getenv("CONTACT_LOG", "/tmp/contact.jsonl"), "a") as f: f.write(json.dumps({**m.model_dump(), "ts": time.time()}) + "\n")
    return {"ok": True}

@app.get("/metrics", response_class=PlainTextResponse)
def metrics(): return "".join(f"exomind_{k} {v}\n" for k, v in M.items())


class Batch(BaseModel):
    items: list[TCE] = Field(max_length=50)

@app.post("/api/v1/batch")
def batch(b: Batch):
    M["batch_items_total"] += len(b.items)
    return [{"kepid": t.kepid, "p_planet": round(score(t)[0], 4)} for t in b.items]

@app.get("/api/v1/rag/search")
def rag_search(q: str = Query(min_length=2, max_length=200)): return bm25(q, 5)

@app.get("/api/v1/model")
def model_card():
    return {"name": "ExoNet-V2", "inputs": {"global_view": "(B,1,2001)", "local_view": "(B,1,201)", "metadata": "(B,3)"},
            "split": "GroupShuffleSplit by kepid, 80/10/10", "leakage_guard": "autovetter columns (av_pred_class, av_pp_*) rejected",
            "tracking": ["train_loss", "roc_auc", "pr_auc"], "export": "ONNX", "serving_mode": "heuristic-demo (swap score() for ONNX)"}

class Label(BaseModel):
    kepid: int
    label: str = Field(pattern="^(planet|false_positive)$")

@app.post("/api/v1/hitl/label")
def hitl_label(l: Label):
    for i in HITL:
        if i["kepid"] == l.kepid: i["label"] = l.label
    M["hitl_labels_total"] += 1
    return {"ok": True}

@app.get("/api/v1/stats")
def stats(): return dict(M) | {"hitl_pending": sum(1 for i in HITL if not i.get("label"))}

for _d in (BASE / "public", BASE.parent / "public"):  # local/Docker only; Vercel serves public/ from its CDN
    if _d.exists():
        app.mount("/", StaticFiles(directory=_d, html=True), name="site"); break

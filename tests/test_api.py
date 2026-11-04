import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "backend"))
from fastapi.testclient import TestClient
from main import app
c = TestClient(app)
P = dict(period=12.4, duration=3.2, depth_ppm=1200, snr=28, secondary_ppm=0)
def test_health(): assert c.get("/api/v1/health").json()["status"] == "ok"
def test_clean_signal_is_candidate(): assert c.post("/api/v1/predict", json=P).json()["label"] == "candidate"
def test_binary_flagged():
    r = c.post("/api/v1/agent/analyze", json=dict(period=2.2, duration=4, depth_ppm=25000, snr=60, secondary_ppm=15000)).json()
    assert "secondary eclipse" in " ".join(r["flags"]) and r["rag_context"]
def test_validation(): assert c.post("/api/v1/predict", json={**P, "period": -1}).status_code == 422
def test_contact(): assert c.post("/api/v1/contact", json={"name": "A", "email": "a@b.co", "message": "hello"}).json()["ok"]

def test_model_card(): assert c.get("/api/v1/model").json()["name"] == "ExoNet-V2"
def test_rag(): assert c.get("/api/v1/rag/search", params={"q": "secondary eclipse"}).json()
def test_batch(): assert len(c.post("/api/v1/batch", json={"items": [P, P]}).json()) == 2

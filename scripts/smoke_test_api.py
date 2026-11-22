"""Quick smoke test — POST /score against the running Hypercorn server."""
import json
import urllib.request
from pathlib import Path

essay = Path("tests/golden/grade_6_argumentative/essay_proficient.txt").read_text()
payload = json.dumps({"grade": 6, "genre": "argumentative", "essay_text": essay}).encode()

req = urllib.request.Request(
    "http://localhost:8000/score",
    data=payload,
    headers={"Content-Type": "application/json"},
)
r = urllib.request.urlopen(req, timeout=60)
data = json.loads(r.read())

print(f"POST /score -> HTTP 200")
print(f"total: {data['total']} | level: {data['performance_level']}")
for c in data["criteria"]:
    print(f"  {c['criterion_name']}: {c['level']}/4")

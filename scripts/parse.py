import sys, glob, os, re, csv
from bs4 import BeautifulSoup
from datetime import datetime, timezone, timedelta
src, out = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(os.path.join(src, "messages*.html")), key=lambda p: int(re.sub(r"\D","",os.path.basename(p)) or 1))
rows = []
for fp in files:
    soup = BeautifulSoup(open(fp, encoding="utf-8").read(), "html.parser")
    for m in soup.select("div.message.default"):
        d = m.select_one("div.date.details")
        tx = m.select_one("div.text")
        if not d or not tx: continue
        ts = datetime.strptime(d["title"], "%d.%m.%Y %H:%M:%S UTC%z").astimezone(timezone.utc)
        for br in tx.find_all("br"): br.replace_with(" ")
        text = re.sub(r"\s+", " ", tx.get_text()).strip()
        if text: rows.append((ts.strftime("%Y-%m-%d %H:%M:%S"), m.get("id",""), text))
rows.sort()
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["utc","id","text"]); w.writerows(rows)
print(len(rows), rows[0][0], rows[-1][0])

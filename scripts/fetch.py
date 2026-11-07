import json, time, urllib.request, sys, csv
OUT = sys.argv[1]
start = 1765440300000 - 60000*5
now = int(time.time()*1000)
rows = []
t = start
while t < now:
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol=XAUUSDT&interval=1m&startTime={t}&limit=1500"
    for a in range(5):
        try:
            d = json.load(urllib.request.urlopen(url, timeout=30)); break
        except Exception as e:
            print("retry", e, file=sys.stderr); time.sleep(2+a*3)
    if not d: break
    rows.extend(d)
    t = d[-1][0] + 60000
    time.sleep(0.12)
with open(OUT, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["ts","open","high","low","close","volume"])
    for r in rows: w.writerow([r[0], r[1], r[2], r[3], r[4], r[5]])
print("rows", len(rows))

"""Live SMC state scan for XAUUSDT: HTF structure, liquidity map, HTF POIs, LTF setups forming now.
usage: python3 smc_live.py data/xauusdt_1m_binance.csv"""
import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from smc_backtest import load, bars, pivots, structure_bias, run, prep, in_kz, NY

px = load(sys.argv[1])["2026-09-15":]
now = px.index[-1] + pd.Timedelta(minutes=1)
price = px.close.iloc[-1]
nyt = lambda t: pd.Timestamp(t).tz_convert(NY).strftime("%a %H:%M")
ist = lambda t: (pd.Timestamp(t).tz_convert("Asia/Kolkata")).strftime("%H:%M")
print(f"NOW {now.strftime('%Y-%m-%d %H:%M')} UTC | {nyt(now)} NY | {ist(now)} IST | price {price:.2f} | killzone: {in_kz(now) or 'none'}")

# ---------- HTF structure ----------
print("\n== HTF STRUCTURE")
TFS = {"1D": ("1D", 2), "4h": ("4h", 2), "1h": ("1h", 3), "15m": ("15min", 3)}
htf = {}
for name, (rule, k) in TFS.items():
    b = bars(px, rule)
    b = b[b.index + pd.Timedelta(rule) <= now]           # closed bars only
    SH, SL = pivots(b, k)
    st = structure_bias(b, k)
    state = int(st.iloc[-1])
    # when did the state last flip, and at what level
    flips = st[st.diff() != 0]
    flip_t = flips.index[-1]
    sh = [s for s in SH if s[0] <= now]; sl = [s for s in SL if s[0] <= now]
    lsh, lsl = sh[-1], sl[-1]
    rng_hi, rng_lo = max(lsh[1], b.high.loc[lsl[2]:].max()), lsl[1]
    eq = (lsh[1] + lsl[1]) / 2
    zone = "PREMIUM" if price > eq else "DISCOUNT"
    htf[name] = dict(state=state, sh=lsh, sl=lsl, eq=eq)
    print(f"{name:>4}: {'BULLISH' if state==1 else 'BEARISH'} since {nyt(flip_t)} NY | last swing H {lsh[1]:.2f} ({nyt(lsh[2])}) | last swing L {lsl[1]:.2f} ({nyt(lsl[2])}) | EQ {eq:.2f} -> price in {zone}")

# ---------- liquidity map ----------
P = prep(px)
day = now.tz_convert(NY).date()
lv = [(n, p, av, sd, sw) for n, p, av, sd, sw in P["levels"].get(day, []) if av <= now]
liq = []
for n, p, av, sd, sw in lv:
    liq.append((n, p, "SWEPT " + nyt(sw) if sw < now else "open"))
for tag, src in (("1h swing H", P["sh1"]), ("1h swing L", P["sl1"])):
    for ct, p, sw in src:
        if now - pd.Timedelta(days=4) <= ct <= now and sw >= now: liq.append((tag, p, "open"))
b4 = bars(px, "4h"); SH4, SL4 = pivots(b4[b4.index + pd.Timedelta("4h") <= now], 2)
hi1m, lo1m = px.high, px.low
for tag, src, side in (("4h swing H", SH4, 1), ("4h swing L", SL4, -1)):
    for ct, p, bt in src[-8:]:
        later = px.loc[ct:]
        taken = (later.high.max() >= p) if side == 1 else (later.low.min() <= p)
        if not taken: liq.append((tag, p, "open"))
wk = px.loc[pd.Timestamp("2026-11-02", tz=NY).tz_convert("UTC"):]
liq += [("Week high", wk.high.max(), "current"), ("Week low", wk.low.min(), "current")]
# equal highs/lows on 15m (two swings within $1.5)
b15 = bars(px.loc[now - pd.Timedelta(days=2):], "15min"); SH15, SL15 = pivots(b15, 3)
for tag, src, side in (("EQH 15m", SH15, 1), ("EQL 15m", SL15, -1)):
    for i in range(len(src)):
        for j in range(i + 1, len(src)):
            if abs(src[i][1] - src[j][1]) <= 1.5:
                p = max(src[i][1], src[j][1]) if side == 1 else min(src[i][1], src[j][1])
                later = px.loc[src[j][0]:]
                if (side == 1 and later.high.max() < p) or (side == -1 and later.low.min() > p):
                    liq.append((tag, p, f"open ({nyt(src[i][2])} & {nyt(src[j][2])})"))
L = pd.DataFrame(liq, columns=["level", "price", "status"]).drop_duplicates(subset=["level", "price"])
L["dist"] = L.price - price
print("\n== LIQUIDITY MAP (nearest first)")
above = L[(L.dist > 0)].sort_values("dist"); below = L[(L.dist < 0)].sort_values("dist", ascending=False)
print(" ABOVE (buy-side):"); [print(f"   {r.price:9.2f}  +{r.dist:6.2f}  {r.level:<11} {r.status}") for r in above.head(8).itertuples()]
print(" BELOW (sell-side):"); [print(f"   {r.price:9.2f}  {r.dist:7.2f}  {r.level:<11} {r.status}") for r in below.head(8).itertuples()]

# ---------- HTF POIs: unmitigated FVGs + last order block on 4h/1h ----------
print("\n== HTF POIs within $60")
for name, rule in (("4h", "4h"), ("1h", "1h"), ("15m", "15min")):
    b = bars(px.loc[now - pd.Timedelta(days=10):], rule); b = b[b.index + pd.Timedelta(rule) <= now]
    h, l, idx = b.high.values, b.low.values, b.index
    for i in range(1, len(b) - 1):
        for kind, bot, top in (("bull FVG", h[i - 1], l[i + 1]), ("bear FVG", h[i + 1], l[i - 1])):
            if top <= bot: continue
            after = px.loc[idx[i + 1] + pd.Timedelta(rule):]
            if kind == "bull FVG":
                filled_to = after.low.min() if len(after) else top
                if filled_to <= bot: continue
                rem_top = min(top, filled_to)
                if abs(price - rem_top) > 60 or rem_top > price: continue
                print(f" {name:>3} {kind}: {bot:.2f} – {rem_top:.2f} (CE {(bot+rem_top)/2:.2f}) formed {nyt(idx[i])} NY {'· partly filled' if rem_top < top else ''}")
            else:
                filled_to = after.high.max() if len(after) else bot
                if filled_to >= top: continue
                rem_bot = max(bot, filled_to)
                if abs(price - rem_bot) > 60 or rem_bot < price: continue
                print(f" {name:>3} {kind}: {rem_bot:.2f} – {top:.2f} (CE {(rem_bot+top)/2:.2f}) formed {nyt(idx[i])} NY {'· partly filled' if rem_bot > bot else ''}")

# ---------- LTF: setups forming today ----------
print("\n== LTF SETUP SCAN (today, NY day)")
start = now.tz_convert(NY).normalize().tz_localize(None).strftime("%Y-%m-%d %H:%M")
end = (now + pd.Timedelta(minutes=1)).tz_convert(NY).tz_localize(None).strftime("%Y-%m-%d %H:%M")
for ltf in ("1min", "3min", "5min"):
    for bias in ("4h", None):
        tr, st = run(px, start, end, ltf=ltf, use_bias=bias, use_kz=True, entry="ce", tp_mode="liq")
        bm = int(pd.Timedelta(ltf).total_seconds() // 60)
        for s in st:
            t = pd.Timestamp(s["time"])
            age = (now - t).total_seconds() / 60
            o = s.get("outcome", "")
            res = s.get("result")
            live = ""
            if o.startswith("no MSS") and age <= 60 + 15: live = "  <-- LIVE: sweep done, waiting for MSS"
            if o == "order" and (res is None or res == "OPEN (marked)" or (isinstance(res, str) and res.startswith("not filled (expired)") and age < 90)): live = "  <-- LIVE ORDER/POSITION"
            print(f" [{ltf[:-3]}m {'4h-bias' if bias else 'no-bias'}] {nyt(t)} NY  swept {s.get('level')} {s.get('price', '')} -> {s.get('dir', '?')}: {o}"
                  + (f" | entry {s['entry']} SL {s['sl']} TP {s['tp']} ({s.get('tp_name')}) -> {res} {s.get('R_net','')}" if o == "order" else "") + live)

# ---------- raw LTF read: last 2h 5m structure ----------
b5 = bars(px.loc[now - pd.Timedelta(hours=8):], "5min"); b5 = b5[b5.index + pd.Timedelta("5min") <= now]
SH5, SL5 = pivots(b5, 3)
print("\n== 5m swings (last 8h):")
print("   highs:", [(nyt(s[2])[4:], round(s[1], 2)) for s in SH5[-5:]])
print("   lows: ", [(nyt(s[2])[4:], round(s[1], 2)) for s in SL5[-5:]])
st5 = structure_bias(bars(px.loc[now - pd.Timedelta(days=2):], "5min"), 3)
print("   5m structure:", "BULLISH" if st5.iloc[-1] == 1 else "BEARISH", "| last flip", nyt(st5[st5.diff() != 0].index[-1]))
r = px.loc[now - pd.Timedelta(minutes=30):]
print(f"   last 30m: H {r.high.max():.2f} L {r.low.min():.2f} | last 5 1m closes {[round(x,2) for x in px.close.iloc[-5:]]}")

import sys, os, json, re, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from classify import classify, SCHEDULED
PX, NEWS, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(7)

# ---------- price ----------
px = pd.read_csv(PX)
px["t"] = pd.to_datetime(px.ts, unit="ms", utc=True)
px = px.set_index("t")[["open","high","low","close","volume"]].astype(float)
px = px[~px.index.duplicated()]
full = pd.date_range(px.index[0], px.index[-1], freq="1min")
traded = px.reindex(full).close.notna()
px = px.reindex(full)
px["close"] = px.close.ffill()
for c in ["open","high","low"]: px[c] = px[c].fillna(px.close)
px["volume"] = px.volume.fillna(0)
px["live"] = traded.values
o, h, l, c = px.open.values, px.high.values, px.low.values, px.close.values
N = len(px)
print("1m bars", N, "live", int(traded.sum()), px.index[0], px.index[-1])

HZ = {"1m":1, "5m":5, "15m":15, "1h":60, "4h":240, "1d":1440}
# forward window stats starting at each minute (anchor = open of that minute)
fw = {}
for k, m in HZ.items():
    hi = pd.Series(h).rolling(m, min_periods=1).max().shift(-(m-1)).values
    lo = pd.Series(l).rolling(m, min_periods=1).min().shift(-(m-1)).values
    cl = pd.Series(c).shift(-(m-1)).values
    R = (hi - lo) / o
    win = max(10080, 20*m)
    base = pd.Series(R).where(px.live.values).rolling(win, min_periods=1440).median().shift(m).values
    # same minute-of-day baseline: median of R at this UTC minute over prior 20 days
    Rs = pd.Series(R, index=px.index)
    mat = Rs.groupby([Rs.index.normalize(), Rs.index.hour*60+Rs.index.minute]).first().unstack()
    tod = mat.rolling(20, min_periods=8).median().shift(1).stack().reindex(pd.MultiIndex.from_arrays([Rs.index.normalize(), Rs.index.hour*60+Rs.index.minute])).values
    fw[k] = dict(hi=hi, lo=lo, cl=cl, R=R, base=base, tod=tod)

# ATR per timeframe (Wilder 14) on resampled bars, mapped back to each minute (value known at bar open)
TF_RULE = {"1m":"1min","5m":"5min","15m":"15min","1h":"1h","4h":"4h","1d":"1D"}
bars, atr_at_min = {}, {}
for k, rule in TF_RULE.items():
    b = px.resample(rule, label="left", closed="left").agg(open=("open","first"),high=("high","max"),low=("low","min"),close=("close","last"),volume=("volume","sum"),live=("live","max"))
    b = b[b.live > 0].drop(columns="live")
    pc = b.close.shift()
    tr = pd.concat([b.high-b.low, (b.high-pc).abs(), (b.low-pc).abs()], axis=1).max(axis=1)
    b["atr"] = tr.ewm(alpha=1/14, adjust=False).mean().shift()   # ATR known before bar opens
    per_day = {"1m":1440,"5m":288,"15m":96,"1h":24,"4h":6,"1d":1}[k]
    b["rng_atr"] = (b.high-b.low)/b.atr
    b["rng_rel"] = ((b.high-b.low)/(b.high-b.low).rolling(max(7*per_day, 20), min_periods=max(2*per_day,10)).median().shift()).replace([np.inf,-np.inf], np.nan)
    body = (b.close-b.open).abs()
    b["wick_frac"] = 1 - body/(b.high-b.low).replace(0,np.nan)
    bars[k] = b
    atr_at_min[k] = b.atr.reindex(px.index, method="ffill").values

# ---------- news ----------
news = pd.read_csv(NEWS)
news["t"] = pd.to_datetime(news.utc, utc=True)
news = news[(news.t >= px.index[0] + pd.Timedelta(days=1)) & (news.t <= px.index[-1] - pd.Timedelta(minutes=5))].copy()
news["cat"] = news.text.map(classify)
news["text"] = news.text.str.replace(r"\s*\(@WalterBloomberg\)\s*$", "", regex=True)
PRIORITY = ["FOMC decision / Powell presser","US CPI/PPI inflation","US jobs (NFP/claims/JOLTS/ADP)","US growth data (GDP/ISM/retail/PMI)",
 "Fed chair / Fed independence politics","Iran / Middle East war","Tariffs / trade war","US-China","Russia / Ukraine","Venezuela / LatAm / other geo",
 "Fed speakers","Gold/CB fundamentals (PBOC, ETF, CB buying)","US fiscal / Treasury / shutdown / debt","Dollar / FX","Other central banks (ECB/BOJ/BOE/SNB)",
 "Oil / energy","Trump (other statements)","Crypto","Equities / earnings / corporate","Other / misc","Gold price report (lagging)"]
prank = {k:i for i,k in enumerate(PRIORITY)}
news["prio"] = news.cat.map(prank)
news = news.sort_values("t").reset_index(drop=True)
news.to_csv(os.path.join(OUT, "news_classified_xau_period.csv"), index=False, columns=["utc","cat","text"])

# cluster headlines into events: new event if > 3 min since the event's first headline
eid, start, ids = -1, None, []
for t in news.t:
    if start is None or (t - start) > pd.Timedelta(minutes=3):
        eid += 1; start = t
    ids.append(eid)
news["eid"] = ids
def agg(g):
    g2 = g.sort_values(["prio","t"])
    return pd.Series(dict(t=g.t.min(), cat=g2.cat.iloc[0], n_head=len(g), headline=g2.text.iloc[0][:300],
                          all_heads=" || ".join(g.text.str[:200].tolist()[:6])))
ev = news.groupby("eid").apply(agg, include_groups=False).reset_index()
ev["scheduled"] = ev.cat.isin(SCHEDULED)
print("events", len(ev))

idx0 = px.index.get_indexer(ev.t.dt.floor("1min")) - 1   # anchor: open of minute before headline minute
ev = ev[(idx0 > 30) & (idx0 < N-1441)].copy(); idx0 = idx0[(idx0 > 30) & (idx0 < N-1441)]
ev["i0"] = idx0
ev["live"] = px.live.values[idx0] | px.live.values[np.minimum(idx0+1,N-1)]
ev = ev[ev.live].copy()

def measure(i0):
    P0 = o[i0]; out = {"p0": P0}
    for k, m in HZ.items():
        f = fw[k]
        up = f["hi"][i0]/P0-1; dn = f["lo"][i0]/P0-1
        out[f"ret_{k}"] = f["cl"][i0]/P0-1
        out[f"mult_{k}"] = f["R"][i0]/f["base"][i0] if f["base"][i0] > 0 else np.nan
        out[f"todmult_{k}"] = f["R"][i0]/f["tod"][i0] if f["tod"][i0] > 0 else np.nan
        out[f"whip_{k}"] = min(up,-dn)/max(up,-dn) if max(up,-dn) > 0 else 0   # 0 = clean one-way, 1 = both sides swept equally
        atr = atr_at_min[k][i0]
        # "double sweep": a 1xATR stop on BOTH sides gets hit within the next 3 bars of that TF
        m3 = min(3*m, N-i0-1)
        hh = h[i0:i0+m3].max(); ll = l[i0:i0+m3].min()
        out[f"dsweep_{k}"] = float(hh >= P0+atr and ll <= P0-atr) if atr > 0 else np.nan
        out[f"range_atr_{k}"] = (hh-ll)/atr if atr > 0 else np.nan
    # impulse fade: direction of first 5 min reversed by 60 min close
    r5, r60 = out["ret_5m"], out["ret_1h"]
    out["fade"] = float(np.sign(r5) != np.sign(r60) and abs(r5) > 0)
    pr = (h[i0-6:i0].max()-l[i0-6:i0].min())/o[i0-6]
    out["pre_mult_5m"] = pr/fw["5m"]["base"][i0] if fw["5m"]["base"][i0] > 0 else np.nan
    out["move15_usd"] = (fw["15m"]["hi"][i0]-fw["15m"]["lo"][i0])
    return out
M = pd.DataFrame([measure(i) for i in ev.i0], index=ev.index)
ev = pd.concat([ev, M], axis=1)

# baseline: random live minutes
live_idx = np.where(px.live.values[:N-1441])[0]; live_idx = live_idx[live_idx > 1500]
base_ix = rng.choice(live_idx, 20000, replace=False)
B = pd.DataFrame([measure(i) for i in base_ix])
# baseline at same minute-of-day (controls for 13:30 UTC etc.)
mod = pd.Series(px.index[base_ix].hour*60 + px.index[base_ix].minute)

def summarize(d):
    r = {"events": len(d)}
    for k in HZ:
        r[f"mult_{k}"] = d[f"mult_{k}"].median()
        r[f"dsweep_{k}"] = d[f"dsweep_{k}"].mean()
        r[f"whip_{k}"] = d[f"whip_{k}"].median()
        r[f"todmult_{k}"] = d[f"todmult_{k}"].median()
        r[f"shock_{k}"] = (d[f"mult_{k}"] >= 3).mean()
    r["fade"] = d.fade.mean(); r["move15_usd"] = d.move15_usd.median()
    r["p90_mult_15m"] = d["mult_15m"].quantile(.9)
    r["pre_mult_5m"] = d["pre_mult_5m"].median()
    return pd.Series(r)
cat_sum = ev.groupby("cat").apply(summarize, include_groups=False)
cat_sum.loc["— Random baseline (no news filter)"] = summarize(B)
cat_sum = cat_sum.sort_values("mult_15m", ascending=False)
cat_sum.to_csv(os.path.join(OUT, "category_impact_summary.csv"))

# rank events by trade-hostility: abnormal range on 15m & 1h + two-sided sweep
ev["hostility"] = (np.log1p(ev.mult_5m.clip(0,50)) + np.log1p(ev.mult_15m.clip(0,50)) + np.log1p(ev.mult_1h.clip(0,50))) * (0.5 + ev.whip_1h.fillna(0)) 
ev["date"] = ev.t.dt.strftime("%Y-%m-%d %H:%M")
cols = ["date","cat","scheduled","n_head","headline","p0","move15_usd","pre_mult_5m"] + [f"mult_{k}" for k in HZ] + [f"todmult_{k}" for k in HZ] + [f"ret_{k}" for k in HZ] + [f"whip_{k}" for k in HZ] + [f"dsweep_{k}" for k in HZ] + ["fade","hostility","all_heads"]
ev.sort_values("t")[cols].to_csv(os.path.join(OUT, "news_events_impact.csv"), index=False)

# ---------- spike candles per TF -> news attribution ----------
LEAD = {"1m":2, "5m":3, "15m":5, "1h":10, "4h":20, "1d":0}
LAG = {"1m":3, "5m":3, "15m":5, "1h":5, "4h":0, "1d":0}
spike_tables, spike_stats = {}, {}
nt = news.t.values
IMPACTFUL = set(PRIORITY[:17]) - {"Gold price report (lagging)"}
for k, b in bars.items():
    b = b.dropna(subset=["atr","rng_rel"]).copy()
    b = b[b.index >= px.index[0] + pd.Timedelta(days=2)]
    thr = b.rng_rel.quantile(0.99 if k in ("1m","5m","15m") else 0.95)
    sp = b[b.rng_rel >= max(thr, 1.8 if k in ("4h","1d") else 2.0)].copy()
    whip = b[(b.rng_rel >= 2.0) & (b.wick_frac >= 0.6)]
    dur = pd.Timedelta(TF_RULE[k])
    def attrib(t0):
        a = np.datetime64(t0 - pd.Timedelta(minutes=LEAD[k])); z = np.datetime64(t0 + dur + pd.Timedelta(minutes=LAG[k]))
        lo_, hi_ = np.searchsorted(nt, a), np.searchsorted(nt, z)
        g = news.iloc[lo_:hi_]
        g = g[g.cat.isin(IMPACTFUL)].sort_values(["prio","t"])
        return g
    rows = []
    for t0, r in sp.iterrows():
        g = attrib(t0)
        rows.append(dict(bar_open=t0.strftime("%Y-%m-%d %H:%M"), range_usd=r.high-r.low, range_x_normal=r.rng_rel, range_x_atr=r.rng_atr, ret_pct=(r.close/r.open-1)*100,
                         wick_frac=r.wick_frac, news_cat=g.cat.iloc[0] if len(g) else "NO NEWS FOUND",
                         top_headline=g.text.iloc[0][:250] if len(g) else "", n_relevant_heads=len(g)))
    st = pd.DataFrame(rows, columns=["bar_open","range_usd","range_x_normal","range_x_atr","ret_pct","wick_frac","news_cat","top_headline","n_relevant_heads"])
    # base rate: how often a random bar of this TF contains impactful news in the same attribution window
    samp = b.sample(min(3000, len(b)), random_state=1)
    base_rate = np.mean([len(attrib(t0)) > 0 for t0 in samp.index])
    wrows = [len(attrib(t0)) > 0 for t0 in whip.index]
    spike_stats[k] = dict(bars=len(b), spikes=len(sp), threshold_x_normal=float(max(thr,2.0)),
                          pct_spikes_with_news=float((st.news_cat != "NO NEWS FOUND").mean()) if len(st) else 0,
                          base_rate_news=float(base_rate), whipsaw_bars=len(whip),
                          pct_whipsaw_with_news=float(np.mean(wrows)) if wrows else 0)
    st = st.sort_values("range_x_normal", ascending=False)
    if len(st): spike_hours = pd.to_datetime(st.bar_open).dt.hour.value_counts().sort_index().to_dict(); spike_stats[k]["by_hour_utc"] = {int(x):int(y) for x,y in spike_hours.items()}
    st.to_csv(os.path.join(OUT, f"spikes_{k}.csv"), index=False)
    spike_tables[k] = st
    print(k, spike_stats[k])

# ---------- seasonality ----------
lv = px[px.live]
r1 = ((lv.high - lv.low)/lv.open*1e4)
hr = r1.groupby([lv.index.dayofweek, lv.index.hour]).mean().unstack()
mins = r1.groupby(lv.index.hour*60 + lv.index.minute).mean()
news_hr = news[news.cat.isin(IMPACTFUL)].groupby(news.t.dt.hour).size()
days = max(1,(px.index[-1]-px.index[0]).days)

# ---------- export for report ----------
srt = ev[~ev.cat.isin(["Equities / earnings / corporate","Other / misc","Crypto"])].sort_values("hostility", ascending=False); keep = []
for i, r in srt.iterrows():
    if all(abs((r.t - ev.at[j,"t"]).total_seconds()) > 5400 for j in keep): keep.append(i)
    if len(keep) >= 40: break
top = ev.loc[keep]
def chart(i0):
    s, e = i0-30, i0+180
    seg = px.iloc[s:e]
    return [[int(t.timestamp()), round(r.open,2), round(r.high,2), round(r.low,2), round(r.close,2)] for t, r in seg.iterrows()]
def clean(x):
    if isinstance(x, (float,np.floating)): return None if not np.isfinite(x) else round(float(x),5)
    if isinstance(x,(np.integer,)): return int(x)
    if isinstance(x,(np.bool_,)): return bool(x)
    return x
top_ev = [{**{k:clean(v) for k,v in r[cols].items()}, "anchor": int(px.index[r.i0].timestamp()), "candles": chart(r.i0)} for _, r in top.iterrows()]
# daily candles + daily news-event intensity
d1 = bars["1d"]
ev_day = ev.groupby(ev.t.dt.floor("D")).agg(n=("hostility","size"), host=("hostility","max"))
daily = [[int(t.timestamp()), round(r.open,2), round(r.high,2), round(r.low,2), round(r.close,2), round(float(r.rng_atr),2) if np.isfinite(r.rng_atr) else None] for t, r in d1.iterrows()]
# monthly category mix of high-impact events (mult_15m >= 3)
hi_ev = ev[ev.mult_15m >= 3]
monthly = hi_ev.groupby([hi_ev.t.dt.strftime("%Y-%m"), "cat"]).size().unstack(fill_value=0)
# minute-of-day around key releases
key_minutes = {f"{m//60:02d}:{m%60:02d}": round(float(v),2) for m, v in mins.items()}
payload = dict(
  meta=dict(start=str(px.index[0]), end=str(px.index[-1]), n_news=int(len(news)), n_events=int(len(ev)), n_1m=int(px.live.sum()),
            last_price=float(c[-1])),
  cat_summary=[{"cat":i, **{k:clean(v) for k,v in r.items()}} for i, r in cat_sum.iterrows()],
  spike_stats=spike_stats,
  spikes={k: [{kk:clean(vv) for kk,vv in r.items()} for _, r in v.head(25).iterrows()] for k, v in spike_tables.items()},
  spike_nonews_by_hour={k: (lambda s: {int(x):int(y) for x,y in pd.to_datetime(s[s.news_cat=="NO NEWS FOUND"].bar_open).dt.hour.value_counts().sort_index().items()})(v) for k,v in spike_tables.items() if len(v)},
  spike_cat_mix={k: v.news_cat.value_counts().to_dict() for k, v in spike_tables.items()},
  top_events=top_ev,
  heat=dict(rows=[[None if not np.isfinite(x) else round(float(x),2) for x in hr.loc[d].reindex(range(24)).values] for d in hr.index], days=[int(d) for d in hr.index]),
  news_by_hour={int(k): round(v/days,2) for k,v in news_hr.items()},
  minute_of_day=key_minutes,
  daily=daily,
  monthly=dict(months=list(monthly.index), cats=list(monthly.columns), data=monthly.values.tolist()),
  sched_vs_unsched={ "scheduled": summarize(ev[ev.scheduled]).map(clean).to_dict(), "unscheduled_geo_politics": summarize(ev[ev.cat.isin(["Iran / Middle East war","Tariffs / trade war","Russia / Ukraine","US-China","Venezuela / LatAm / other geo","Fed chair / Fed independence politics","Trump (other statements)"])]).map(clean).to_dict(), "baseline": summarize(B).map(clean).to_dict()},
)
json.dump(payload, open(os.path.join(OUT, "report_data.json"), "w"), default=str)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(cat_sum[["events","mult_1m","mult_5m","mult_15m","mult_1h","mult_4h","dsweep_5m","dsweep_15m","dsweep_1h","fade","move15_usd"]].round(2))
print(ev.sort_values("hostility", ascending=False).head(25)[["date","cat","mult_5m","mult_15m","mult_1h","whip_1h","move15_usd","headline"]].to_string(max_colwidth=90))

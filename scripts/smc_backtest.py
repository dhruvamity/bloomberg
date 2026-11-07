"""
Mechanical SMC / ICT backtest on Binance XAUUSDT 1m data.

Model ("liquidity sweep -> market structure shift -> FVG entry"):
  1. HTF bias: 4h market structure (last close beyond a confirmed 4h swing high/low).
  2. Liquidity: previous-day high/low (NY day, weekdays), weekend high/low (Mondays),
     Asian range (20:00-00:00 NY), London range (02:00-05:00 NY), unswept confirmed 1h swings.
  3. Sweep: a 5m bar trades through a level and a 5m bar closes back inside within 3 bars,
     inside a killzone (London 02:00-05:00 NY, NY AM 07:00-10:00 NY).
  4. MSS: within 12 5m bars, a 5m close beyond the last confirmed 5m swing (opposite side),
     with displacement leaving a fair value gap in the leg.
  5. Entry: limit at the FVG (CE = 50% or proximal edge), valid 60 min; SL beyond the sweep
     extreme + 0.1 ATR(5m); TP = nearest opposing liquidity (min 2R) or fixed R.
  6. One position at a time; flat by 16:00 NY. Fill/exit resolved on 1m, SL-first if ambiguous.
No lookahead: swings only exist after their right-side bars close; ranges only after they end.
"""
import sys, os, json
import numpy as np, pandas as pd

NY = "America/New_York"
COST_PCT = 0.0005   # round-trip fees+slippage as fraction of price (maker in / taker out, approx)

def load(path):
    px = pd.read_csv(path)
    px["t"] = pd.to_datetime(px.ts, unit="ms", utc=True)
    return px.set_index("t")[["open", "high", "low", "close"]].astype(float)

def bars(px, rule):
    b = px.resample(rule, label="left", closed="left").agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last")).dropna()
    pc = b.close.shift()
    tr = pd.concat([b.high - b.low, (b.high - pc).abs(), (b.low - pc).abs()], axis=1).max(axis=1)
    b["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    return b

def pivots(b, k):
    """confirmed swing highs/lows: list of (confirm_time, price, bar_time)."""
    h, l, idx = b.high.values, b.low.values, b.index
    dur = idx[1] - idx[0]
    SH, SL = [], []
    for i in range(k, len(b) - k):
        if h[i] >= h[i - k:i].max() and h[i] > h[i + 1:i + k + 1].max():
            SH.append((idx[i + k] + dur, h[i], idx[i]))
        if l[i] <= l[i - k:i].min() and l[i] < l[i + 1:i + k + 1].min():
            SL.append((idx[i + k] + dur, l[i], idx[i]))
    return SH, SL

def structure_bias(b, k):
    """+1 / -1 state after each closed bar, indexed by bar close time."""
    SH, SL = pivots(b, k)
    sh_i = sl_i = 0; last_sh = last_sl = None; state = 0; out = {}
    dur = b.index[1] - b.index[0]
    for t, r in b.iterrows():
        close_t = t + dur
        while sh_i < len(SH) and SH[sh_i][0] <= close_t: last_sh = SH[sh_i][1]; sh_i += 1
        while sl_i < len(SL) and SL[sl_i][0] <= close_t: last_sl = SL[sl_i][1]; sl_i += 1
        if last_sh is not None and r.close > last_sh: state = 1; last_sh = None
        elif last_sl is not None and r.close < last_sl: state = -1; last_sl = None
        out[close_t] = state
    return pd.Series(out)

def session_levels(px):
    """Per NY date: PDH/PDL (previous weekday), weekend H/L (Mon), Asia H/L, London H/L with availability times."""
    ny = px.copy(); ny.index = ny.index.tz_convert(NY)
    d = ny.groupby(ny.index.date).agg(high=("high", "max"), low=("low", "min"))
    d.index = pd.to_datetime(d.index)
    lv = {}
    dates = list(d.index)
    for i, day in enumerate(dates):
        L = []
        day_ny = pd.Timestamp(day).tz_localize(NY)
        # previous weekday
        j = i - 1
        while j >= 0 and dates[j].weekday() >= 5: j -= 1
        if j >= 0:
            L += [("PDH", d.high.iloc[j], day_ny, +1), ("PDL", d.low.iloc[j], day_ny, -1)]
        if day.weekday() == 0:
            wk = ny[(ny.index >= day_ny - pd.Timedelta(days=2)) & (ny.index < day_ny)]
            if len(wk): L += [("WkndH", wk.high.max(), day_ny, +1), ("WkndL", wk.low.min(), day_ny, -1)]
        a0, a1 = day_ny - pd.Timedelta(hours=4), day_ny
        asia = ny[(ny.index >= a0) & (ny.index < a1)]
        if len(asia): L += [("AsiaH", asia.high.max(), a1, +1), ("AsiaL", asia.low.min(), a1, -1)]
        l0, l1 = day_ny + pd.Timedelta(hours=2), day_ny + pd.Timedelta(hours=5)
        lon = ny[(ny.index >= l0) & (ny.index < l1)]
        if len(lon): L += [("LonH", lon.high.max(), l1, +1), ("LonL", lon.low.min(), l1, -1)]
        lv[day.date()] = [(n, p, avail.tz_convert("UTC"), side) for n, p, avail, side in L]
    return lv

def in_kz(t_utc):
    t = t_utc.tz_convert(NY); m = t.hour * 60 + t.minute
    if 120 <= m < 300: return "London"
    if 420 <= m < 600: return "NY AM"
    return None

_CACHE = {}
def prep(px):
    key = (px.index[0], px.index[-1], len(px))
    if key in _CACHE: return _CACHE[key]
    P = dict(b5=bars(px, "5min"), b1h=bars(px, "1h"), b4h=bars(px, "4h"))
    P["bias"] = {"4h": structure_bias(P["b4h"], 2), "1h": structure_bias(P["b1h"], 3)}
    P["levels"] = session_levels(px)
    SH1, SL1 = pivots(P["b1h"], 2)
    hi, lo, idx = px.high.values, px.low.values, px.index
    def swept_time(ct, p, side):
        i = idx.searchsorted(ct); seg = hi[i:i + 7200] if side == 1 else lo[i:i + 7200]
        hit = np.nonzero(seg >= p)[0] if side == 1 else np.nonzero(seg <= p)[0]
        return idx[i + hit[0]] if len(hit) else idx[-1] + pd.Timedelta(days=30)
    P["levels"] = {day: [(n, p, av, sd, swept_time(av, p, sd)) for n, p, av, sd in L] for day, L in P["levels"].items()}
    P["sh1"] = [(ct, p, swept_time(ct, p, 1)) for ct, p, _ in SH1]
    P["sl1"] = [(ct, p, swept_time(ct, p, -1)) for ct, p, _ in SL1]
    _CACHE[key] = P
    return P

def run(px, start, end, entry="ce", tp_mode="liq", use_bias="4h", use_kz=True, min_rr=2.0, max_per_kz=2, ltf="5min", swing_k=3):
    P = prep(px)
    b5 = bars(px, ltf) if ltf != "5min" else P["b5"]
    bm = int(pd.Timedelta(ltf).total_seconds() // 60)
    RECLAIM, MSSW = max(1, 15 // bm), max(3, 60 // bm)
    atr5 = P["b5"].atr.reindex(b5.index, method="ffill")
    b4h = P["b4h"]
    bias4 = P["bias"][use_bias] if use_bias else P["bias"]["4h"]
    bias_idx = bias4.index.values
    SH5, SL5 = pivots(b5, swing_k)
    levels = P["levels"]
    t5 = b5.index; o5, h5, l5, c5 = (b5[c].values for c in ["open", "high", "low", "close"]); a5 = atr5.values
    m1 = px; m1_idx = m1.index
    start = pd.Timestamp(start, tz=NY).tz_convert("UTC"); end = pd.Timestamp(end, tz=NY).tz_convert("UTC")
    i0 = t5.searchsorted(start); i1 = t5.searchsorted(end)
    trades, setups = [], []
    busy_until = start; used = set(); kz_count = {}
    sh5_t = np.array([s[0].value for s in SH5]); sl5_t = np.array([s[0].value for s in SL5])
    sh1, sl1 = P["sh1"], P["sl1"]

    def bias_at(t):
        k = bias4.index.searchsorted(t, side="right") - 1
        return int(bias4.iloc[k]) if k >= 0 else 0

    def liq_levels(t, side):
        """available opposing/target levels at time t: side=+1 buy-side (above), -1 sell-side (below)."""
        day = t.tz_convert(NY).date()
        out = [(n, p) for n, p, av, sd, sw in levels.get(day, []) if sd == side and av <= t and sw >= t]
        src = sh1 if side == 1 else sl1
        lo_t = t - pd.Timedelta(days=5)
        for ct, p, sw in src:
            if lo_t <= ct <= t and sw >= t:
                out.append(("1hSH" if side == 1 else "1hSL", p))
        return out

    i = i0
    while i < i1 - 2:
        t = t5[i]
        kz = in_kz(t)
        if (use_kz and kz is None) or t < busy_until:
            i += 1; continue
        key_kz = (t.tz_convert(NY).date(), kz)
        if kz_count.get(key_kz, 0) >= max_per_kz:
            i += 1; continue
        found = False
        for side in (-1, 1):            # side of liquidity swept: -1 = sell-side (long setup), +1 = buy-side (short)
            for name, L in liq_levels(t, side):
                if (t.tz_convert(NY).date(), name, round(L, 2)) in used: continue
                breach = (l5[i] < L) if side == -1 else (h5[i] > L)
                if not breach: continue
                # already beyond before this bar? then it's not a fresh sweep
                if (side == -1 and c5[i - 1] < L) or (side == 1 and c5[i - 1] > L): continue
                # reclaim within 3 bars
                rj = None
                for j in range(i, min(i + RECLAIM, i1)):
                    if (side == -1 and c5[j] > L) or (side == 1 and c5[j] < L): rj = j; break
                used.add((t.tz_convert(NY).date(), name, round(L, 2)))
                if rj is None:
                    setups.append(dict(time=str(t), level=name, price=L, outcome="no reclaim (real breakout)")); continue
                d = 1 if side == -1 else -1      # trade direction
                ext = l5[i:rj + 1].min() if d == 1 else h5[i:rj + 1].max()
                b = bias_at(t5[rj] + pd.Timedelta(minutes=bm))
                rec = dict(time=str(t), kz=kz, level=name, price=round(L, 2), dir="LONG" if d == 1 else "SHORT", bias=b)
                if use_bias and b != d:
                    rec["outcome"] = "against 4h bias"; setups.append(rec); continue
                # MSS reference: last confirmed 5m swing (opposite side) before reclaim close
                tc = (t5[rj] + pd.Timedelta(minutes=bm)).value
                if d == 1:
                    k = np.searchsorted(sh5_t, tc, side="right") - 1
                    if k < 0: continue
                    mss_lvl = SH5[k][1]
                else:
                    k = np.searchsorted(sl5_t, tc, side="right") - 1
                    if k < 0: continue
                    mss_lvl = SL5[k][1]
                m = None
                for j in range(rj, min(rj + MSSW, i1 - 1)):
                    if (d == 1 and l5[j] < ext) or (d == -1 and h5[j] > ext): break
                    if (d == 1 and c5[j] > mss_lvl) or (d == -1 and c5[j] < mss_lvl): m = j; break
                if m is None:
                    rec["outcome"] = "no MSS (sweep failed to shift structure)"; setups.append(rec); continue
                # FVGs in the leg
                fvg = None
                for q in range(i + 1, m + 2):
                    if q + 1 >= len(t5): break
                    if d == 1 and l5[q + 1] > h5[q - 1]: fvg = (h5[q - 1], l5[q + 1])       # (bottom, top)
                    if d == -1 and h5[q + 1] < l5[q - 1]: fvg = (h5[q + 1], l5[q - 1])
                if fvg is None:
                    rec["outcome"] = "MSS without displacement/FVG"; setups.append(rec); continue
                bot, top = fvg
                ep = (bot + top) / 2 if entry == "ce" else (top if d == 1 else bot)
                atr = a5[m]
                sl = ext - 0.1 * atr if d == 1 else ext + 0.1 * atr
                risk = (ep - sl) * d
                place_t = t5[m + 1] + pd.Timedelta(minutes=bm)
                if risk <= 0 or risk > 4 * atr or risk < 0.25 * atr:
                    rec["outcome"] = f"risk out of bounds ({risk:.2f})"; setups.append(rec); continue
                if (d == 1 and c5[m + 1] <= ep) or (d == -1 and c5[m + 1] >= ep):
                    rec["outcome"] = "FVG already traded through"; setups.append(rec); continue
                if tp_mode == "liq":
                    cands = [(n2, p2) for n2, p2 in liq_levels(place_t, 1 if d == 1 else -1) if (p2 - ep) * d >= min_rr * risk]
                    near = [(n2, p2) for n2, p2 in liq_levels(place_t, 1 if d == 1 else -1) if (p2 - ep) * d > 0]
                    if not near:
                        rec["outcome"] = "no opposing liquidity"; setups.append(rec); continue
                    n2, tp = min(near, key=lambda x: (x[1] - ep) * d)
                    if (tp - ep) * d < min_rr * risk:
                        rec["outcome"] = f"nearest target {n2} only {((tp-ep)*d/risk):.1f}R"; setups.append(rec); continue
                    tp_name = n2
                else:
                    tp = ep + d * tp_mode * risk; tp_name = f"{tp_mode}R"
                # simulate on 1m
                day_ny = t.tz_convert(NY).normalize()
                flat_t = (day_ny + pd.Timedelta(hours=16)).tz_convert("UTC")
                expire = place_t + pd.Timedelta(minutes=60)
                seg = m1.loc[place_t:flat_t]
                fill_t = None; exit_t = None; exit_p = None; res = None
                for tt, r in seg.iterrows():
                    if fill_t is None:
                        if tt >= expire: res = "not filled (expired)"; exit_t = tt; break
                        if (d == 1 and r.high >= tp) or (d == -1 and r.low <= tp): res = "not filled (ran to target)"; exit_t = tt; break
                        if (d == 1 and r.low <= ep) or (d == -1 and r.high >= ep):
                            fill_t = tt
                            if (d == 1 and r.low <= sl) or (d == -1 and r.high >= sl): res = "LOSS"; exit_t = tt; exit_p = sl; break
                        continue
                    if (d == 1 and r.low <= sl) or (d == -1 and r.high >= sl): res = "LOSS"; exit_t = tt; exit_p = sl; break
                    if (d == 1 and r.high >= tp) or (d == -1 and r.low <= tp): res = "WIN"; exit_t = tt; exit_p = tp; break
                if fill_t is not None and res is None:
                    last = seg.iloc[-1]; exit_t = seg.index[-1]; exit_p = last.close
                    res = "TIME EXIT" if exit_t >= flat_t - pd.Timedelta(minutes=1) else "OPEN (marked)"
                rec.update(outcome="order", entry=round(ep, 2), sl=round(sl, 2), tp=round(tp, 2), tp_name=tp_name, risk=round(risk, 2),
                           placed=str(place_t), mss_level=round(mss_lvl, 2), sweep_ext=round(ext, 2))
                if fill_t is None:
                    rec["result"] = res; setups.append(rec)
                    busy_until = exit_t if exit_t is not None else expire
                else:
                    R = (exit_p - ep) * d / risk
                    cost = COST_PCT * ep / risk
                    rec.update(result=res, filled=str(fill_t), exit=str(exit_t), exit_price=round(exit_p, 2), R=round(R, 2), R_net=round(R - cost, 2),
                               mfe_R=round(((seg.loc[fill_t:exit_t].high.max() - ep) if d == 1 else (ep - seg.loc[fill_t:exit_t].low.min())) / risk, 2),
                               mae_R=round(((ep - seg.loc[fill_t:exit_t].low.min()) if d == 1 else (seg.loc[fill_t:exit_t].high.max() - ep)) / risk, 2))
                    trades.append(rec); setups.append(rec)
                    busy_until = exit_t
                    kz_count[key_kz] = kz_count.get(key_kz, 0) + 1
                found = True
                break
            if found: break
        i += 1
    return trades, setups

if __name__ == "__main__":
    px = load(sys.argv[1])
    start, end = sys.argv[2], sys.argv[3]
    tr, st = run(px, start, end)
    print(json.dumps(tr, indent=1, default=str))

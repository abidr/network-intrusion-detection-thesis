"""
TTL join: attach sttl/dttl from the original UNSW-NB15 full flow files to every
row of CICFlowMeter_out.csv (same 2015 traffic, two extractors).

Match key: 5-tuple (srcip, sport, dstip, dsport, proto) in both orientations
(reversed match swaps sttl/dttl), then time: prefer candidates whose
[Stime-TOL, Ltime+TOL] window contains the CFM flow start, else nearest Stime
within NEAR_TOL. The CFM->epoch offset is estimated empirically from
single-candidate exact matches (expected ~+14400s, Atlantic time).

Fallback layer: per-(srcip,dstip) modal (sttl,dttl) for rows with no flow match
(TTL is mostly a host-pair property).

Outputs (results/ttl_join/):
  cache_ttl_raw.npz  -- per raw CFM row (3,540,241): sttl, dttl (int16, -1 = missing),
                        match_type (0 none / 1 direct / 2 reversed / 3 ip-pair)
  ttl_join.txt/.json -- coverage + the FINGERPRINT CHECK: per-class sttl distributions.
GO/NO-GO gate for the pipeline experiment -- no model is trained here.
"""
import io, json, sys, time, gc
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np

ROOT = Path(r"E:\AIUB\THESIS")
CFM = ROOT / "Resource" / "CICFlowMeter_out.csv"
UDIR = ROOT / "Resource" / "UNSW-NB15"
OUT = Path(__file__).parent
TOL, NEAR_TOL = 5, 60
LOG = []


def say(s=""):
    print(s); sys.stdout.flush(); LOG.append(str(s))


def dump():
    io.open(OUT / "ttl_join.txt", "w", encoding="utf-8").write("\n".join(LOG))


PROTO = {"tcp": 6, "udp": 17, "icmp": 1}


def ip32(s):
    a, b, c, d = s.split(".")
    return (int(a) << 24) | (int(b) << 16) | (int(c) << 8) | int(d)


def port_i(s):
    try:
        return int(s)
    except ValueError:
        try:
            return int(s, 16) & 0xFFFF
        except ValueError:
            return -1


def key_of(si, sp, di, dp, pr):
    # si:32 sp:16 | di:32 dp:16 pr:8 -> 104 bits, fields non-overlapping
    return (((si << 16) | sp) << 56) | (((di << 16) | dp) << 8) | pr


t_start = time.time()
say("=" * 74); say("1. INDEX UNSW-NB15 FULL FILES"); say("=" * 74)

flows = defaultdict(list)          # key -> list of (stime, ltime, sttl, dttl)
pair_ttl = defaultdict(Counter)    # (srcip,dstip) -> Counter of (sttl,dttl)
n_u = n_skip = 0
for fn in ("UNSW-NB15_1.csv", "UNSW-NB15_2.csv", "UNSW-NB15_3.csv", "UNSW-NB15_4.csv"):
    t0 = time.time()
    with open(UDIR / fn, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            p = line.split(",", 12)
            if len(p) < 12:
                n_skip += 1; continue
            pr = PROTO.get(p[4])
            if pr is None:
                n_skip += 1; continue
            sp, dp = port_i(p[1]), port_i(p[3])
            if sp < 0 or dp < 0:
                n_skip += 1; continue
            try:
                si, di = ip32(p[0]), ip32(p[2])
                sttl, dttl = int(p[9]), int(p[10])
            except ValueError:
                n_skip += 1; continue
            # stime/ltime are cols 28/29 -- re-split the tail only when needed
            q = line.rsplit(",", 21)
            stime, ltime = int(q[-21]), int(q[-20])
            flows[key_of(si, sp, di, dp, pr)].append((stime, ltime, sttl, dttl))
            pair_ttl[(si, di)][(sttl, dttl)] += 1
            n_u += 1
    say("  %s done (%.0fs, cumulative %d flows)" % (fn, time.time() - t0, n_u))
say("indexed %d UNSW flows | %d unique 5-tuples | %d skipped (non-tcp/udp/icmp or unparsable)"
    % (n_u, len(flows), n_skip))
pair_mode = {k: c.most_common(1)[0][0] for k, c in pair_ttl.items()}
del pair_ttl; gc.collect()
dump()

say(""); say("=" * 74); say("2. PARSE CFM IDENTITY + ESTIMATE TIME OFFSET"); say("=" * 74)

_daycache = {}


def cfm_epoch(ts):
    # "22/01/2015 07:50:15 AM" -> naive epoch (as if UTC)
    date, clock, ap = ts.split(" ")
    base = _daycache.get(date)
    if base is None:
        dd, mm, yy = date.split("/")
        days = (np.datetime64("%s-%s-%s" % (yy, mm, dd)) - np.datetime64("1970-01-01")
                ).astype(int)
        base = int(days) * 86400
        _daycache[date] = base
    hh, mi, ss = clock.split(":")
    h = int(hh) % 12
    if ap == "PM":
        h += 12
    return base + h * 3600 + int(mi) * 60 + int(ss)


# pass over CFM once, storing parsed identity compactly
N = 3540241
KEYD = [0] * N; KEYR = [0] * N          # 104-bit ints -- python list, not numpy
TS = np.zeros(N, np.int64)
SIP = np.zeros(N, np.uint32); DIP = np.zeros(N, np.uint32)
lab_list = []
t0 = time.time()
i = 0
with open(CFM, "r", encoding="utf-8", errors="replace") as f:
    f.readline()
    for line in f:
        c = line.rstrip("\r\n")
        if not c:
            continue
        p = c.split(",", 8)
        si, di = ip32(p[1]), ip32(p[3])
        sp, dp = int(p[2]), int(p[4]); pr = int(p[5])
        KEYD[i] = key_of(si, sp, di, dp, pr)
        KEYR[i] = key_of(di, dp, si, sp, pr)
        TS[i] = cfm_epoch(p[6])
        SIP[i], DIP[i] = si, di
        lab_list.append(c.rsplit(",", 1)[1])
        i += 1
assert i == N, i
say("CFM parsed %d rows (%.0fs)" % (i, time.time() - t0))

# offset from single-candidate direct matches
offs = []
for j in range(0, N, 37):                       # ~96k sample
    cand = flows.get(int(KEYD[j]))
    if cand is not None and len(cand) == 1:
        offs.append(cand[0][0] - int(TS[j]))
    if len(offs) >= 40000:
        break
offs = np.array(offs)
OFFSET = int(np.median(offs))
say("offset sample n=%d | median %+d s | within +/-2s of median: %.1f%%"
    % (len(offs), OFFSET, 100 * (np.abs(offs - OFFSET) <= 2).mean()))
dump()

say(""); say("=" * 74); say("3. MATCH"); say("=" * 74)


def pick(cand, t):
    best, bd = None, None
    for stime, ltime, st_, dt_ in cand:
        if stime - TOL <= t <= ltime + TOL:
            d = abs(stime - t)
            if bd is None or d < bd:
                best, bd = (st_, dt_), d
    if best is not None:
        return best
    for stime, ltime, st_, dt_ in cand:
        d = abs(stime - t)
        if d <= NEAR_TOL and (bd is None or d < bd):
            best, bd = (st_, dt_), d
    return best


STTL = np.full(N, -1, np.int16)
DTTL = np.full(N, -1, np.int16)
MTYPE = np.zeros(N, np.int8)
t0 = time.time()
for j in range(N):
    t = int(TS[j]) + OFFSET
    cand = flows.get(int(KEYD[j]))
    if cand is not None:
        r = pick(cand, t)
        if r is not None:
            STTL[j], DTTL[j] = r; MTYPE[j] = 1
            continue
    cand = flows.get(int(KEYR[j]))
    if cand is not None:
        r = pick(cand, t)
        if r is not None:
            DTTL[j], STTL[j] = r; MTYPE[j] = 2
            continue
    pm = pair_mode.get((int(SIP[j]), int(DIP[j])))
    if pm is None:
        pm = pair_mode.get((int(DIP[j]), int(SIP[j])))
        if pm is not None:
            STTL[j], DTTL[j] = pm[1], pm[0]; MTYPE[j] = 3
    else:
        STTL[j], DTTL[j] = pm; MTYPE[j] = 3
say("matched in %.0fm" % ((time.time() - t0) / 60))
np.savez_compressed(OUT / "cache_ttl_raw.npz", sttl=STTL, dttl=DTTL, mtype=MTYPE)

lab = np.array(lab_list); del lab_list, flows, pair_mode; gc.collect()
say("match type: direct %.2f%% | reversed %.2f%% | ip-pair %.2f%% | NONE %.2f%%"
    % tuple(100 * (MTYPE == k).mean() for k in (1, 2, 3, 0)))
dump()

say(""); say("=" * 74); say("4. COVERAGE + FINGERPRINT CHECK (per class)"); say("=" * 74)
res_cls = {}
say("%-16s %9s | flow-match | any-match | top sttl values (share)" % ("class", "rows"))
for cls in sorted(set(lab.tolist())):
    m = lab == cls
    n = int(m.sum())
    fm = float(((MTYPE == 1) | (MTYPE == 2))[m].mean())
    am = float((MTYPE[m] > 0).mean())
    have = m & (STTL >= 0)
    top = Counter(STTL[have].tolist()).most_common(4)
    tops = ", ".join("%d(%.0f%%)" % (v, 100 * c / max(1, int(have.sum()))) for v, c in top)
    say("%-16s %9d |   %6.2f%% |  %6.2f%% | %s" % (cls, n, 100 * fm, 100 * am, tops))
    res_cls[cls] = dict(rows=n, flow_match=round(fm, 4), any_match=round(am, 4),
                        top_sttl={str(v): round(c / max(1, int(have.sum())), 3) for v, c in top})

# fingerprint severity: how well does sttl ALONE predict benign-vs-attack?
have_all = STTL >= 0
yb = (lab != "Benign").astype(int)
best_acc = {}
for v in np.unique(STTL[have_all]):
    mv = have_all & (STTL == v)
    if mv.sum() < 1000:
        continue
    best_acc[int(v)] = float(yb[mv].mean())
say("\nattack share by sttl value (>=1000 rows) -- values near 0 or 1 mean sttl acts as a label proxy:")
for v, a in sorted(best_acc.items()):
    say("  sttl %3d : %8d rows, attack share %.4f" % (v, int((have_all & (STTL == v)).sum()), a))

res = dict(n_unsw=n_u, unsw_skipped=n_skip, offset_s=OFFSET,
           match_pct=dict(direct=round(100 * float((MTYPE == 1).mean()), 2),
                          reversed=round(100 * float((MTYPE == 2).mean()), 2),
                          ippair=round(100 * float((MTYPE == 3).mean()), 2),
                          none=round(100 * float((MTYPE == 0).mean()), 2)),
           per_class=res_cls,
           attack_share_by_sttl={str(k): round(v, 4) for k, v in best_acc.items()},
           minutes=round((time.time() - t_start) / 60, 1))
(OUT / "ttl_join.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
say("\ntotal %.1f min" % ((time.time() - t_start) / 60))
dump()

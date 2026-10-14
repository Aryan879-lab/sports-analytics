"""Data Science in Sports - Cricket implementation (simulated T20 data)."""
import os, json
import numpy as np, pandas as pd
SHOW = True   # True = open each chart in a window (close it to see the next). False = just save PNGs.
import matplotlib
if not SHOW: matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, log_loss, brier_score_loss

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "output") + os.sep      # figures + CSVs are saved here
os.makedirs(OUT, exist_ok=True)
def finish():
    plt.show() if SHOW else plt.close()

rng = np.random.default_rng(42)
sns.set_theme(style="whitegrid")
NAVY = "#1F4E79"; ORANGE = "#D9822B"; GREY = "#8A8F98"
R = {}

# ---------- Part A: batting performance data ----------
roles = {"Opener": (0.9, 135), "Middle order": (1.0, 128), "Finisher": (0.7, 150), "All-rounder": (0.75, 125)}
rows = []
for pid in range(1, 61):
    role = list(roles)[pid % 4]
    skill = rng.normal(1.0, 0.15)
    for inn in range(1, 31):
        balls = max(1, int(rng.gamma(2.2, 9) * roles[role][0]))
        sr = max(40, rng.normal(roles[role][1] * skill, 25))
        runs = int(round(balls * sr / 100))
        rows.append((pid, role, inn, balls, runs, runs / balls * 100))
bat = pd.DataFrame(rows, columns=["player_id", "role", "innings_no", "balls_faced", "runs", "strike_rate"])
bat.to_csv(OUT + "batting_innings.csv", index=False)
R["bat_rows"] = len(bat); R["bat_players"] = bat.player_id.nunique()
R["bat_desc"] = bat[["balls_faced", "runs", "strike_rate"]].describe().round(2).to_dict()
role_tbl = bat.groupby("role").agg(innings=("runs", "size"), avg_runs=("runs", "mean"), avg_balls=("balls_faced", "mean"),
                                   strike_rate=("strike_rate", "mean")).round(2)
R["role_tbl"] = role_tbl.reset_index().to_dict("records")
# 6-innings rolling strike rate for one player
p = bat[bat.player_id == 8].sort_values("innings_no").tail(6)
R["p8"] = p[["innings_no", "runs", "balls_faced", "strike_rate"]].round(1).to_dict("records")
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.plot(range(1, 7), p.strike_rate.values, marker="o", color=NAVY, lw=2)
ax.set(xlabel="Innings (most recent six)", ylabel="Strike rate", title="Strike Rate Trend - Last 6 Innings (Player 8)")
ax.set_xticks(range(1, 7)); fig.tight_layout(); fig.savefig(OUT + "f_trend.png", dpi=200); finish()
fig, ax = plt.subplots(figsize=(6.4, 3.8))
sns.boxplot(data=bat, x="role", y="strike_rate", color="#9DB8D2", ax=ax)
ax.set(xlabel="", ylabel="Strike rate", title="Strike Rate Distribution by Batting Role"); fig.tight_layout()
fig.savefig(OUT + "f_role.png", dpi=200); finish()
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(bat[["balls_faced", "runs", "strike_rate"]].corr(), annot=True, fmt=".2f", cmap="Blues", ax=ax)
ax.set_title("Correlation of Batting Metrics"); fig.tight_layout(); fig.savefig(OUT + "f_corr.png", dpi=200); finish()
R["corr_balls_runs"] = round(bat.balls_faced.corr(bat.runs), 2)
R["corr_sr_runs"] = round(bat.strike_rate.corr(bat.runs), 2)

# ---------- Part B: ball-by-ball chase simulation ----------
P_RUN = np.array([0.40, 0.37, 0.07, 0.01, 0.09, 0.06])  # 0,1,2,3,4,6
RUNS = np.array([0, 1, 2, 3, 4, 6])

def ball(s, over, wk):
    agg = 1 + 0.04 * max(0, over - 14) + (0.03 if over < 6 else 0)       # death / powerplay
    agg *= s
    pw = 0.035 * (1 + 0.1 * (wk >= 6)) * (1.15 if over >= 15 else 1) / s
    if rng.random() < pw: return -1
    pr = P_RUN.copy(); pr[4:] *= agg; pr[:2] *= (2 - agg) if agg < 2 else 0.1; pr /= pr.sum()
    return rng.choice(RUNS, p=pr)

def innings(s, target=None):
    runs = wk = 0; hist = []
    for b in range(120):
        ov = b // 6
        if target is not None:
            hist.append((b, runs, wk))
        r = ball(s, ov, wk)
        if r < 0: wk += 1
        else: runs += r
        if wk == 10 or (target is not None and runs >= target): 
            return runs, wk, b + 1, hist
    return runs, wk, 120, hist

rows = []
for m in range(2000):
    s1, s2 = rng.normal(1.0, 0.1, 2)
    t1, _, _, _ = innings(s1); target = t1 + 1
    r2, w2, balls2, hist = innings(s2, target)
    win = int(r2 >= target)
    for b, runs, wk in hist:
        rows.append((m, target, b, runs, wk, win))
df = pd.DataFrame(rows, columns=["match_id", "target", "balls_bowled", "runs", "wickets_lost", "chase_won"])
df["runs_left"] = df.target - df.runs; df["balls_left"] = 120 - df.balls_bowled; df["wickets_left"] = 10 - df.wickets_lost
df["crr"] = np.where(df.balls_bowled > 0, df.runs * 6 / df.balls_bowled.clip(lower=1), 0)
df["rrr"] = df.runs_left * 6 / df.balls_left.clip(lower=1)
df = df[df.balls_left > 0]
df.to_csv(OUT + "chase_states.csv", index=False)
R["n_matches"] = int(df.match_id.nunique()); R["n_states"] = len(df)
R["win_rate"] = round(df.groupby("match_id").chase_won.first().mean() * 100, 2)
R["target_desc"] = df.groupby("match_id").target.first().describe().round(1).to_dict()
FEATS = ["runs_left", "balls_left", "wickets_left", "crr", "rrr", "target"]
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr, te = next(gss.split(df, groups=df.match_id))
Xtr, Xte, ytr, yte = df.iloc[tr][FEATS], df.iloc[te][FEATS], df.iloc[tr].chase_won, df.iloc[te].chase_won
R["train_matches"] = int(df.iloc[tr].match_id.nunique()); R["test_matches"] = int(df.iloc[te].match_id.nunique())
R["train_rows"], R["test_rows"] = len(tr), len(te)
lr = LogisticRegression(max_iter=2000).fit((Xtr - Xtr.mean()) / Xtr.std(), ytr)
lrp = lr.predict_proba((Xte - Xtr.mean()) / Xtr.std())[:, 1]
rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=50, random_state=42, n_jobs=-1).fit(Xtr, ytr)
rfp = rf.predict_proba(Xte)[:, 1]
def rep(name, p):
    pred = (p >= .5).astype(int)
    return dict(model=name, accuracy=round(accuracy_score(yte, pred) * 100, 2), auc=round(roc_auc_score(yte, p), 4),
                logloss=round(log_loss(yte, p), 4), brier=round(brier_score_loss(yte, p), 4))
R["models"] = [rep("Logistic Regression", lrp), rep("Random Forest", rfp)]
pred = (rfp >= .5).astype(int)
R["cm"] = confusion_matrix(yte, pred).tolist()
R["cls"] = classification_report(yte, pred, target_names=["Chase lost", "Chase won"], output_dict=True)
R["imp"] = sorted(zip(FEATS, rf.feature_importances_.round(4).tolist()), key=lambda x: -x[1])
# accuracy by phase of chase
te_df = df.iloc[te].copy(); te_df["ok"] = (pred == yte.values); te_df["phase"] = pd.cut(te_df.balls_bowled, [-1, 35, 71, 107, 120],
    labels=["Overs 1-6", "Overs 7-12", "Overs 13-18", "Overs 19-20"])
R["phase_acc"] = (te_df.groupby("phase", observed=True).ok.mean() * 100).round(2).to_dict()
R["phase_acc"] = {str(k): v for k, v in R["phase_acc"].items()}
# figures
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(R["cm"], annot=True, fmt="d", cmap="Blues", xticklabels=["Lost", "Won"], yticklabels=["Lost", "Won"], ax=ax)
ax.set(xlabel="Predicted", ylabel="Actual", title="Confusion Matrix - Random Forest"); fig.tight_layout(); fig.savefig(OUT + "f_cm.png", dpi=200); finish()
fig, ax = plt.subplots(figsize=(6.4, 3.8))
names, vals = zip(*R["imp"][::-1]); ax.barh(names, vals, color=NAVY)
ax.set(xlabel="Importance", title="Random Forest Feature Importance"); fig.tight_layout(); fig.savefig(OUT + "f_imp.png", dpi=200); finish()
# win probability curve for a test match
mid = df.iloc[te].match_id.unique()
def curve(m):
    d = df[df.match_id == m]; return d, rf.predict_proba(d[FEATS])[:, 1]
pick = None
for m in mid:
    d, pr = curve(m)
    if len(d) > 90 and d.chase_won.iloc[0] == 1 and pr.min() < .25 and pr.max() > .9 and 0.3 < pr[0] < .7: pick = m; break
if pick is None: pick = mid[0]
d, pr = curve(pick)
R["curve_match"] = dict(target=int(d.target.iloc[0]), won=int(d.chase_won.iloc[0]), balls=len(d), start=round(pr[0] * 100, 1), end=round(pr[-1] * 100, 1))
R["curve_pts"] = {int(b): round(float(x) * 100, 1) for b, x in zip(d.balls_bowled, pr) if b in (0, 30, 60, 90, 105)}
fig, ax = plt.subplots(figsize=(6.4, 3.8))
ax.plot(d.balls_bowled / 6, pr * 100, color=NAVY, lw=2, label="Chasing team"); ax.plot(d.balls_bowled / 6, 100 - pr * 100, color=ORANGE, lw=2, label="Defending team")
ax.axhline(50, color=GREY, ls="--", lw=1); ax.set(xlabel="Overs completed", ylabel="Win probability (%)", ylim=(0, 100), title=f"Live Win Probability (target {R['curve_match']['target']})")
ax.legend(); fig.tight_layout(); fig.savefig(OUT + "f_wp.png", dpi=200); finish()
# distribution of target + outcome
fig, ax = plt.subplots(figsize=(6.4, 3.6)); t = df.groupby("match_id").agg(target=("target", "first"), won=("chase_won", "first"))
sns.histplot(data=t, x="target", hue="won", bins=25, multiple="stack", palette=[ORANGE, NAVY], ax=ax)
ax.set(xlabel="Target set by first innings", title="Targets and Chase Outcome (1 = chase won)"); fig.tight_layout(); fig.savefig(OUT + "f_target.png", dpi=200); finish()
# sample prediction
ex = pd.DataFrame([dict(runs_left=45, balls_left=30, wickets_left=6, crr=0, rrr=0, target=170)])
ex["crr"] = (170 - 45) * 6 / 90; ex["rrr"] = 45 * 6 / 30
R["sample"] = dict(row=ex.iloc[0].round(2).to_dict(), p=round(float(rf.predict_proba(ex[FEATS])[0, 1]) * 100, 1))
json.dump(R, open(OUT + "results.json", "w"), indent=1, default=str)
print(json.dumps(R, indent=1, default=str))

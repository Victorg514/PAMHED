"""
Draws the pipeline / model-architecture flowchart for the paper.

  python ml/analysis/flowchart.py   ->  results/figures/pipeline_flowchart.png and .pdf

Layout units are 1/100 inch with y pointing down, so positions read like pixel coordinates.
"""
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch

OUT = pathlib.Path(__file__).resolve().parents[2] / "results" / "figures"

W, H = 1820, 2290
CX = 870                  # main column centre
BW = 600                  # main box width
FILL, EDGE = "#E8F1FB", "#2B63B0"
TITLE_C, SUB_C, ARROW_C = "#0B3A75", "#2B63B0", "#8A8A8A"
plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]

fig = plt.figure(figsize=(W / 100, H / 100))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(H, 0)
ax.axis("off")


def box(x0, y0, w, h, title, subs=(), radius=22, dashed=False, title_size=24, sub_size=19):
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle=f"round,pad=0,rounding_size={radius}",
                                fc=FILL, ec=EDGE, lw=1.6, ls=(0, (6, 4)) if dashed else "-"))
    lines = [(title, title_size, TITLE_C, "bold")] + [(s, sub_size, SUB_C, "normal") for s in subs]
    step = 42
    y = y0 + h / 2 - step * (len(lines) - 1) / 2
    for text, size, color, weight in lines:
        ax.text(x0 + w / 2, y, text, ha="center", va="center", fontsize=size, color=color, fontweight=weight)
        y += step
    return x0, y0, w, h


def stadium(cy, text, w=310, h=95):
    return box(CX - w / 2, cy - h / 2, w, h, text, radius=h / 2)


def datafile(y_mid, text, x0=1240, w=540, h=92, skew=40):
    pts = [(x0 + skew, y_mid - h / 2), (x0 + w, y_mid - h / 2), (x0 + w - skew, y_mid + h / 2), (x0, y_mid + h / 2)]
    ax.add_patch(Polygon(pts, closed=True, fc=FILL, ec=EDGE, lw=1.6))
    ax.text(x0 + w / 2, y_mid, text, ha="center", va="center", fontsize=20, color=SUB_C)


def arrow(p, q, dashed=False, head=True):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="->" if head else "-", mutation_scale=28, lw=2.4,
                                 color=ARROW_C, shrinkA=0, shrinkB=0, ls=(0, (6, 4)) if dashed else "-"))


def line(p, q):
    ax.plot([p[0], q[0]], [p[1], q[1]], color=ARROW_C, lw=2.4, solid_capstyle="butt")


def main_box(y0, h, title, subs=()):
    return box(CX - BW / 2, y0, BW, h, title, subs)


def down(b_from, b_to):
    arrow((CX, b_from[1] + b_from[3]), (CX, b_to[1]))


def to_file(b, text):
    y_mid = b[1] + b[3] / 2
    datafile(y_mid, text)
    arrow((b[0] + b[2], y_mid), (1240 + 20, y_mid))


# --- Legend ---
ax.add_patch(FancyBboxPatch((40, 45), 420, 345, boxstyle="round,pad=0,rounding_size=18", fc="#F9F9F8", ec="#D0D0D0", lw=1.2))
ax.text(90, 100, "Shapes", fontsize=24, va="center", color="black")
legend_y = [160, 220, 280, 340]
ax.add_patch(FancyBboxPatch((80, legend_y[0] - 17), 60, 34, boxstyle="round,pad=0,rounding_size=17", fc=FILL, ec=EDGE, lw=1.4))
ax.add_patch(FancyBboxPatch((80, legend_y[1] - 17), 60, 34, boxstyle="round,pad=0,rounding_size=7", fc=FILL, ec=EDGE, lw=1.4))
ax.add_patch(FancyBboxPatch((80, legend_y[2] - 17), 60, 34, boxstyle="round,pad=0,rounding_size=7", fc=FILL, ec=EDGE, lw=1.4,
                            ls=(0, (4, 3))))
ax.add_patch(Polygon([(92, legend_y[3] - 17), (142, legend_y[3] - 17), (130, legend_y[3] + 17), (80, legend_y[3] + 17)],
                     closed=True, fc=FILL, ec=EDGE, lw=1.4))
for y, label in zip(legend_y, ["Start / end", "Process", "Comparison baseline", "Data file"]):
    ax.text(165, y, label, fontsize=20, va="center", color="#333333")

# --- Data collection and preprocessing ---
start = stadium(92, "Start")
scrape = main_box(190, 140, "Scrape timelines", ["main.py + Nitter"])
images = main_box(380, 140, "Download images", ["download_images.py"])
blip = main_box(570, 140, "BLIP captioning", ["blip_base.py"])
merge = main_box(760, 210, "Merge + feature engineering",
                 ["merge_blip.py", "personal baseline: first 40% of timeline", "deviation features · user-level splits"])
arrow((CX, 140), (CX, scrape[1]))
for a, b in [(scrape, images), (images, blip), (blip, merge)]:
    down(a, b)
to_file(scrape, "user_timelines.json")
to_file(images, "[tweet_id].png")
to_file(blip, "[tweet_id].txt caption")
to_file(merge, "final.parquet")

# --- Model ---
fusion = main_box(1020, 210, "Fusion model (one per task)",
                  ["RoBERTa: tweet text + image caption", "+ 10 behavioural features", "combined by a learned gate"])
down(merge, fusion)
baseline = box(1240, 1065, 540, 120, "Text-only baseline", ["same model, no features"], dashed=True)
arrow((CX + BW / 2, 1125), (1240, 1125), dashed=True, head=False)

# --- Four parallel binary classifiers ---
TASK_W, TASK_H, GAP = 370, 140, 40
task_y = 1340
xs = [CX - (4 * TASK_W + 3 * GAP) / 2 + i * (TASK_W + GAP) for i in range(4)]
tasks = [("Screener", "mental health vs control"), ("Depression", "vs rest"), ("Anxiety", "vs rest"), ("Bipolar", "vs rest")]
centres = [x + TASK_W / 2 for x in xs]
split_y, join_y = 1285, 1535
line((CX, fusion[1] + fusion[3]), (CX, split_y))
line((centres[0], split_y), (centres[-1], split_y))
for x, c, (t, s) in zip(xs, centres, tasks):
    box(x, task_y, TASK_W, TASK_H, t, [s])
    arrow((c, split_y), (c, task_y))
    line((c, task_y + TASK_H), (c, join_y))
line((centres[0], join_y), (centres[-1], join_y))

# --- Prediction ---
threshold = main_box(1590, 140, "Decision threshold", ["tuned on validation (macro F1)"])
arrow((CX, join_y), (CX, threshold[1]))
user = main_box(1780, 140, "User-level score", ["mean probability over a user's tweets"])
down(threshold, user)
shap = main_box(1970, 140, "Prediction + SHAP", ["features and words explained separately"])
down(user, shap)
end = stadium(2210, "End")
arrow((CX, shap[1] + shap[3]), (CX, 2210 - 95 / 2))

OUT.mkdir(parents=True, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(OUT / f"pipeline_flowchart.{ext}", dpi=150, facecolor="white")
print(f"Written to {OUT}")

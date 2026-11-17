
"""Redraw the portfolio charts from the release CSVs."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

INK = "#182822"
SUBTLE = "#445c51"
LINE = "#dbe5dd"
BG = "#f6f8f4"
ACCENT = "#176451"
OTHER = "#8ca99c"

CHARTS = {
    "Simple": {
        "opponent": "Opponent holds near the centre",
        "section": "1,000 games per policy",
        "rows": [
            ("Simple_B0", "evaluation", "B0  ·  Batch 1,024 / width 128"),
            ("Simple_E3", "evaluation", "E3  ·  Batch 256 / width 128"),
            ("Simple_E7", "evaluation", "E7 Final  ·  Batch 256 / width 256"),
        ],
        "finals": {2},
        "footer": ("Same learning rate, 0.0003, for all three.", "Training seed 1. Each policy used fixed spawns."),
    },
    "Medium": {
        "opponent": "Opponent circles the centre",
        "section": "Independent test · 200 new paired starts",
        "rows": [
            ("Eps020_s2", "independent_200", "Eps020_s2  ·  Final, seed 2"),
            ("Medium_M2LR", "independent_200", "M2LR  ·  Previous model"),
            ("Medium_B0", "independent_200", "B0  ·  Baseline"),
        ],
        "finals": {0},
    },
    "Hard": {
        "opponent": "Opponent moves back and forth along an axis",
        "section": "First 100 games per policy",
        "rows": [
            ("Hard_E9_seed2", "first_100", "E9 Final  ·  Linear decay"),
            ("Hard_E11_seed2", "first_100", "E11  ·  Constant learning rate"),
        ],
        "finals": {0},
        "footer": ("Mean reward in last 50k steps: E9 +0.527, E11 +0.576.", "Training seed 2. Random starts were unpaired."),
    },
}

def read_rows(path):
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))

def draw_evaluation(data, output, diagnostics):
    rows = {(r["experiment"],r["stage"]): r for r in data}
    current = diagnostics["Eps020_s2"]
    for name, spec in CHARTS.items():
        fig = plt.figure(figsize=(6.8,5.66), facecolor=BG)
        ax = fig.add_axes([0,0,1,1], xlim=(0,1), ylim=(0,1))
        ax.axis("off")
        ax.text(.065,.92,name,fontsize=31,fontweight="bold",color=INK,va="center")
        ax.text(.066,.853,spec["opponent"],fontsize=11.8,color=SUBTLE,va="center")
        ax.text(.066,.756,spec["section"],fontsize=11.6,color=ACCENT,fontweight="bold")
        positions = {"Simple":[.646,.500,.354], "Medium":[.646,.518,.390], "Hard":[.635,.456]}[name]
        for k,((exp,stage,label),y) in enumerate(zip(spec["rows"],positions)):
            r = rows[(exp,stage)]
            wins,games = int(r["wins"]),int(r["games"])
            final = k in spec["finals"]
            ax.text(.066,y+.038,label,fontsize=10.6,va="center",color=INK,fontweight="bold" if final else "normal")
            ax.text(.934,y+.038,f"{wins:,} / {games:,}",fontsize=10.6,ha="right",va="center",color=INK,fontweight="bold")
            ax.barh(y,.868,left=.066,height=.028,color=LINE,align="center")
            ax.barh(y,.868*wins/games,left=.066,height=.028,color=ACCENT if final else OTHER,align="center")
        scale_y = positions[-1]-.062
        for pct in (0,50,100):
            x = .066+.868*pct/100
            align = "left" if pct==0 else "right" if pct==100 else "center"
            ax.text(x,scale_y,f"{pct}% {'wins' if pct==100 else ''}".strip(),fontsize=8.4,color=SUBTLE,ha=align,va="center")
        if name == "Hard":
            ax.text(.066,.297,"Higher training reward for E11",fontsize=12,fontweight="bold",color=INK)
            ax.text(.066,.245,"E9 won three more games in these evaluations.",fontsize=10.8,color=SUBTLE)
        ax.plot([.066,.934],[.162,.162],color=LINE,lw=1.2)
        if name == "Medium":
            old_selection = rows["Medium_M2LR", "confirmation"]
            old_holdout = rows["Medium_M2LR", "historical_holdout_100"]
            ax.text(.066,.242,
                    f'Final: {current["draws"]} draws ({current["timeouts"]} timeouts), {current["losses"]} losses.',
                    fontsize=10.2,color=INK)
            ax.text(.066,.200,
                    f'Falls: {current["falls_before_contact"]} before contact, {current["falls_after_contact"]} after.',
                    fontsize=10.2,color=INK)
            footer = (f'Previous M2LR selection retest: {old_selection["wins"]}/{old_selection["games"]}.',
                      f'Previous M2LR earlier independent test: {old_holdout["wins"]}/{old_holdout["games"]}.')
        else:
            footer = spec["footer"]
        for j,line in enumerate(footer):
            ax.text(.066,.109-.041*j,line,color=SUBTLE,fontsize=9.1,va="center")
        for ext in ("svg","png"):
            fig.savefig(output/f"{name.lower()}.{ext}",dpi=175,facecolor=BG)
        plt.close(fig)

def draw_training(data, output):
    series = defaultdict(list)
    for row in data:
        series[row["experiment"]].append((int(row["step"]),float(row["cumulative_reward"])))
    pairs = [
        ("Simple","Simple_E7","Simple_B0","E7 Final","B0"),
        ("Medium","Eps020_s2","Medium_B0","Eps020_s2","B0"),
        ("Hard","Hard_E9_seed2","Hard_B0","E9 Final (seed 2)","B0"),
    ]
    serif_fonts = ["Times New Roman","STIX Two Text","DejaVu Serif"]
    with plt.rc_context({"font.family":"serif","font.serif":serif_fonts,"mathtext.fontset":"stix"}):
        fig, axes = plt.subplots(1,3,figsize=(13.8,4.08),sharey=True,facecolor="white")
        fig.subplots_adjust(left=.07,right=.985,bottom=.17,top=.9,wspace=.12)
        fig.text(.5275,.025,"Training steps",color="#000000",fontsize=13,ha="center")
        for i,(ax,(name,final,ref,final_label,ref_label)) in enumerate(zip(axes,pairs)):
            ax.set_facecolor("white")
            for exp,label,color,style in [(ref,ref_label,"#E69F00","--"),(final,final_label,"#0072B2","-")]:
                x,y=zip(*sorted(series[exp]))
                ax.plot(x,y,label=label,color=color,lw=1.9,ls=style,zorder=4)
            ax.set_title(f"({'abc'[i]}) {name}",fontsize=14,color="#000000",loc="left",pad=8)
            for side in ("top","right"):
                ax.spines[side].set_visible(False)
            for side in ("left","bottom"):
                ax.spines[side].set_color("#000000"); ax.spines[side].set_linewidth(.8)
            ax.set_xlim(0,510000)
            ax.set_ylim(-1.26,1.12)
            ax.set_xticks([0,250000,500000])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda tick,pos: "0" if not tick else f"{tick/1000:.0f}k"))
            ax.set_yticks([-1,0,1])
            ax.tick_params(axis="both",colors="#000000",direction="out",length=3.5,width=.8,labelsize=11.5,pad=4)
            if i==0:
                ax.set_ylabel("Cumulative reward",fontsize=13,color="#000000")
            ax.legend(loc="center right" if name == "Hard" else "lower right",frameon=False,fontsize=11,handlelength=2.4)
        fig.savefig(output/"training-figure.png",dpi=110,facecolor="white")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--training",action="store_true",help="Also render the training curve graphic used in the video.")
    options = parser.parse_args()
    data = Path(__file__).resolve().parents[1]/"data"
    options.output.mkdir(parents=True,exist_ok=True)
    diagnostics = {row["model"]: row for row in read_rows(data/"medium_independent_200_summary.csv")}
    draw_evaluation(read_rows(data/"evaluation_summary.csv"),options.output,diagnostics)
    if options.training:
        draw_training(read_rows(data/"training_rewards.csv"),options.output)

if __name__ == "__main__":
    main()


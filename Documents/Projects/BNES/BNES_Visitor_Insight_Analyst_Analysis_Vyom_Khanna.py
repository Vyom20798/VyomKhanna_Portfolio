#!/usr/bin/env python3
"""
Roman Baths Visitor Insight Analysis
April 2025 vs April 2026
Visitor Insight Analyst Assessment
B&NES Heritage Services
Colour scheme: Deep Teal #1A5276, Gold/Amber #D4AC0D, Stone #85929E, White #FFFFFF
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

# === COLOURS (Roman Baths / B&NES brand-inspired) ===
TEAL   = "#1A5276"
GOLD   = "#D4AC0D"
STONE  = "#85929E"
LIGHT  = "#EBF5FB"
WHITE  = "#FFFFFF"
RED    = "#C0392B"
GREEN  = "#1E8449"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": WHITE,
})

# === LOAD DATA ===
df = pd.read_excel("D:/Documents/Job Application Documents/BNES/Visitor_Insights_Analyst_Data_for_Assessment_final.xlsx")
df["Year"] = df["Date"].dt.year
df["DateDay"] = df["Date"].dt.day

# Derived segments
df25 = df[df["Year"] == 2025]
df26 = df[df["Year"] == 2026]

def pct_change(a, b):
    return ((b - a) / a) * 100

# =====================================================
# FIGURE 1: Executive Dashboard (4 KPI tiles + two charts)
# =====================================================
fig1 = plt.figure(figsize=(16, 9), facecolor=WHITE)
fig1.suptitle("Roman Baths Visitor Insight  |  April 2025 vs April 2026",
              fontsize=18, fontweight="bold", color=TEAL, y=0.97)

gs = GridSpec(3, 4, figure=fig1, hspace=0.55, wspace=0.4,
              left=0.06, right=0.97, top=0.90, bottom=0.08)

# ---- KPI Tiles ----
kpis = [
    ("Total Visitors", 92643, 87193, ""),
    ("Total Revenue", 1998784, 2090275, "£"),
    ("Rev / Visitor", 21.58, 23.97, "£"),
    ("Advance Bookings", 12904, 12789, ""),
]
for i, (label, v25, v26, prefix) in enumerate(kpis):
    ax = fig1.add_subplot(gs[0, i])
    ax.set_facecolor(LIGHT)
    ax.axis("off")
    delta = pct_change(v25, v26)
    arrow = "▲" if delta >= 0 else "▼"
    col = GREEN if delta >= 0 else RED
    if prefix == "£" and v25 > 1000:
        fmt25 = f"£{v25/1000:,.0f}K" if v25 < 1e6 else f"£{v25/1e6:,.2f}M"
        fmt26 = f"£{v26/1000:,.0f}K" if v26 < 1e6 else f"£{v26/1e6:,.2f}M"
    elif prefix == "£":
        fmt25 = f"£{v25:.2f}"
        fmt26 = f"£{v26:.2f}"
    else:
        fmt25 = f"{v25:,.0f}"
        fmt26 = f"{v26:,.0f}"
    ax.text(0.5, 0.82, label, ha="center", va="center", fontsize=10,
            color=STONE, transform=ax.transAxes)
    ax.text(0.5, 0.55, fmt26, ha="center", va="center", fontsize=18,
            fontweight="bold", color=TEAL, transform=ax.transAxes)
    ax.text(0.5, 0.25, f"{arrow} {abs(delta):.1f}% vs 2025 ({fmt25})",
            ha="center", va="center", fontsize=9, color=col, transform=ax.transAxes)

# ---- Chart 1: Daily visitors line chart ----
ax1 = fig1.add_subplot(gs[1:, :2])
daily25 = df25.groupby("DateDay")["Qty"].sum().reset_index()
daily26 = df26.groupby("DateDay")["Qty"].sum().reset_index()
ax1.plot(daily25["DateDay"], daily25["Qty"], color=TEAL, lw=2.5, label="April 2025", marker="o", ms=4)
ax1.plot(daily26["DateDay"], daily26["Qty"], color=GOLD, lw=2.5, label="April 2026", marker="o", ms=4, linestyle="--")
# Easter shading
ax1.axvspan(18, 21, alpha=0.12, color=TEAL, label="Easter 2025 (18-21 Apr)")
ax1.axvspan(3, 6, alpha=0.12, color=GOLD, label="Easter 2026 (3-6 Apr)")
ax1.set_xlabel("Day of April", fontsize=10, color=STONE)
ax1.set_ylabel("Daily Visitors", fontsize=10, color=STONE)
ax1.set_title("Daily Visitor Volume", fontsize=12, fontweight="bold", color=TEAL)
ax1.legend(fontsize=8, framealpha=0.5)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax1.set_facecolor(WHITE)
ax1.tick_params(colors=STONE)

# ---- Chart 2: Visitor type breakdown bar ----
ax2 = fig1.add_subplot(gs[1:, 2:])
vtypes = ["Adult", "Child", "Family", "Senior", "Student"]
q25 = [df25[df25["Visitor Type"]==v]["Qty"].sum() for v in vtypes]
q26 = [df26[df26["Visitor Type"]==v]["Qty"].sum() for v in vtypes]
x = np.arange(len(vtypes))
w = 0.35
bars25 = ax2.bar(x - w/2, q25, w, color=TEAL, label="2025")
bars26 = ax2.bar(x + w/2, q26, w, color=GOLD, label="2026")
ax2.set_xticks(x)
ax2.set_xticklabels(vtypes, fontsize=10, color=STONE)
ax2.set_title("Visitors by Type", fontsize=12, fontweight="bold", color=TEAL)
ax2.set_ylabel("Visitors", fontsize=10, color=STONE)
ax2.legend(fontsize=9)
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax2.set_facecolor(WHITE)
ax2.tick_params(colors=STONE)
for bar in bars25:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
             f"{bar.get_height():,.0f}", ha="center", va="bottom", fontsize=7, color=TEAL)
for bar in bars26:
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
             f"{bar.get_height():,.0f}", ha="center", va="bottom", fontsize=7, color="#8B6914")

fig1.savefig("fig1_executive_dashboard.png", dpi=150, bbox_inches="tight")
print("Saved fig1")

# =====================================================
# FIGURE 2: Purchase Behaviour & Advance vs Walk-Up
# =====================================================
fig2, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor=WHITE)
fig2.suptitle("Purchase Behaviour & Channel Shift  |  April 2025 vs April 2026",
              fontsize=16, fontweight="bold", color=TEAL, y=1.01)

# Chart A: Purchase type (excluding Web anomaly - note it)
ptypes = ["Advance", "Walk Up", "Special"]
p25 = [12904, 32850, 8659]
p26 = [12789, 67490, 6572]
x = np.arange(len(ptypes))
ax = axes[0]
ax.bar(x - 0.2, p25, 0.4, color=TEAL, label="2025")
ax.bar(x + 0.2, p26, 0.4, color=GOLD, label="2026")
ax.set_xticks(x); ax.set_xticklabels(ptypes, color=STONE)
ax.set_title("Purchase Channel", fontsize=12, fontweight="bold", color=TEAL)
ax.set_ylabel("Visitors", color=STONE)
ax.legend(); ax.set_facecolor(WHITE); ax.tick_params(colors=STONE)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.text(0.5, -0.15, "* Web channel reclassified in 2026 data", ha="center",
        transform=ax.transAxes, fontsize=8, color=RED, style="italic")

# Chart B: Individual vs Group
ax = axes[1]
cats = ["Individual", "Group"]
v25 = [72038, 20605]
v26 = [69535, 17659]
x = np.arange(2)
ax.bar(x - 0.2, v25, 0.4, color=TEAL, label="2025")
ax.bar(x + 0.2, v26, 0.4, color=GOLD, label="2026")
ax.set_xticks(x); ax.set_xticklabels(cats, color=STONE)
ax.set_title("Individual vs Group", fontsize=12, fontweight="bold", color=TEAL)
ax.set_ylabel("Visitors", color=STONE)
ax.legend(); ax.set_facecolor(WHITE); ax.tick_params(colors=STONE)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))
# Revenue per head annotation
ax.text(0.5, 0.92, "Rev/head 2026: Individual £24.87 | Group £20.45",
        ha="center", transform=ax.transAxes, fontsize=9, color=TEAL,
        bbox=dict(boxstyle="round,pad=0.3", facecolor=LIGHT, edgecolor=TEAL))

# Chart C: Weekday vs Weekend
ax = axes[2]
cats2 = ["Weekday", "Weekend"]
w25 = [64578, 28065]
w26 = [61584, 25609]
x = np.arange(2)
ax.bar(x - 0.2, w25, 0.4, color=TEAL, label="2025")
ax.bar(x + 0.2, w26, 0.4, color=GOLD, label="2026")
ax.set_xticks(x); ax.set_xticklabels(cats2, color=STONE)
ax.set_title("Weekday vs Weekend", fontsize=12, fontweight="bold", color=TEAL)
ax.set_ylabel("Visitors", color=STONE)
ax.legend(); ax.set_facecolor(WHITE); ax.tick_params(colors=STONE)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:,.0f}"))

fig2.tight_layout()
fig2.savefig("fig2_purchase_behaviour.png", dpi=150, bbox_inches="tight")
print("Saved fig2")

# =====================================================
# FIGURE 3: Easter Campaign - Family Focus
# =====================================================
fig3, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=WHITE)
fig3.suptitle("Easter Campaign Impact  |  Family Visitor Analysis",
              fontsize=16, fontweight="bold", color=TEAL, y=1.02)

# Chart A: Family vs Total during Easter windows
ax = axes[0]
windows = ["Full April", "Easter Week"]
fam25 = [17633, 3871]
fam26 = [14742, 4458]
tot25 = [92643, 17619]
tot26 = [87193, 17972]
# Family share
share25 = [f/t*100 for f,t in zip(fam25,tot25)]
share26 = [f/t*100 for f,t in zip(fam26,tot26)]
x = np.arange(2)
ax.bar(x - 0.2, share25, 0.4, color=TEAL, label="2025 Family %")
ax.bar(x + 0.2, share26, 0.4, color=GOLD, label="2026 Family %")
ax.set_xticks(x); ax.set_xticklabels(windows, color=STONE)
ax.set_ylabel("Family Visitors as % of Total", color=STONE)
ax.set_title("Family Share: Full April vs Easter Week", fontsize=12, fontweight="bold", color=TEAL)
ax.legend()
for i, (s25, s26) in enumerate(zip(share25, share26)):
    ax.text(i - 0.2, s25 + 0.2, f"{s25:.1f}%", ha="center", fontsize=11, color=TEAL, fontweight="bold")
    ax.text(i + 0.2, s26 + 0.2, f"{s26:.1f}%", ha="center", fontsize=11, color="#8B6914", fontweight="bold")
ax.set_facecolor(WHITE); ax.tick_params(colors=STONE)
ax.set_ylim(0, 35)

# Chart B: Easter head-to-head
ax = axes[1]
metrics = ["Easter Visitors", "Easter Family", "Easter Rev/head £"]
vals25 = [17619, 3871, 21.58]
vals26 = [17972, 4458, 23.97]
x = np.arange(3)
bars25 = ax.bar(x - 0.2, vals25, 0.4, color=TEAL, label="2025")
bars26 = ax.bar(x + 0.2, vals26, 0.4, color=GOLD, label="2026")
ax.set_xticks(x); ax.set_xticklabels(metrics, color=STONE, fontsize=9)
ax.set_title("Easter Week Head-to-Head", fontsize=12, fontweight="bold", color=TEAL)
ax.legend(); ax.set_facecolor(WHITE); ax.tick_params(colors=STONE)
# Annotate % change
for i, (v2, v1) in enumerate(zip(vals26, vals25)):
    chg = pct_change(v1, v2)
    col = GREEN if chg >= 0 else RED
    ax.text(i + 0.2, v2 + max(vals25)*0.01, f"+{chg:.1f}%" if chg>=0 else f"{chg:.1f}%",
            ha="center", fontsize=10, color=col, fontweight="bold")

fig3.tight_layout()
fig3.savefig("fig3_easter_campaign.png", dpi=150, bbox_inches="tight")
print("Saved fig3")

print("\nAll figures generated successfully.")
print("Key findings summary:")
print(f"  Visitors: 92,643 (2025) -> 87,193 (2026), {pct_change(92643,87193):.1f}%")
print(f"  Revenue:  £1,998,784 (2025) -> £2,090,275 (2026), +{pct_change(1998784,2090275):.1f}%")
print(f"  Rev/visitor: £21.58 -> £23.97, +{pct_change(21.58,23.97):.1f}%")
print(f"  Easter family: 3,871 -> 4,458, +{pct_change(3871,4458):.1f}%")
print(f"  Easter total: 17,619 -> 17,972, +{pct_change(17619,17972):.1f}%")
print(f"  Group visitors: 20,605 -> 17,659, {pct_change(20605,17659):.1f}%")

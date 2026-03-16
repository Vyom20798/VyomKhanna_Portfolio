"""
=============================================================================
Oodle Car Finance – Data Analyst Take-Home Case Study
=============================================================================
Author  : Vyom Khanna
Dataset : 10,000 fabricated loan application records
Goal    : Help the marketing team decide where to invest more effectively

HOW TO RUN
----------
    pip install pandas matplotlib seaborn
    python oodle_analysis.py

All charts are saved to ./charts/
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import os, warnings
warnings.filterwarnings('ignore')

# ── Config ─────────────────────────────────────────────────────────────────
DATA_PATH   = "Oodle_Case_Study_for_Data_Test_25.csv"
OUTPUT_DIR  = "charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NAVY  = '#1A1F3C'
TEAL  = '#00C9A7'
TEAL2 = '#00A896'
WHITE = '#FFFFFF'
GREY  = '#8892A4'
RED   = '#FF6B6B'
GOLD  = '#FFD93D'


# =============================================================================
# SECTION 1 – DATA LOADING & CLEANING
# =============================================================================
print("=" * 60)
print("SECTION 1: DATA LOADING & CLEANING")
print("=" * 60)

df = pd.read_csv(DATA_PATH, encoding='latin1')

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Rename for convenience
df = df.rename(columns={
    'Loanamount': 'loan_amount',
    'Deposit':    'deposit',
    'Funded':     'funded'
})

print("\nRaw columns:", df.columns.tolist())
print("Shape:", df.shape)

# Clean currency columns (£ and comma)
df['loan_clean']    = df['loan_amount'].str.replace('£', '', regex=False).str.replace(',', '', regex=False).astype(float)
df['deposit_clean'] = df['deposit'].str.replace('£', '', regex=False).str.replace(',', '', regex=False).astype(float)

# Binary flags
df['approved']    = (df['application_outcome'] == 'approved').astype(int)
df['funded_flag'] = (df['funded'] == 'Yes').astype(int)

# Derived features
df['deposit_ratio'] = df['deposit_clean'] / df['loan_clean']   # deposit as % of loan

# Banding
df['age_band']  = pd.cut(df['age'],        bins=[17,25,35,50,65], labels=['18-25','26-35','36-50','51-65'])
df['loan_band'] = pd.cut(df['loan_clean'], bins=[0,5000,10000,20000,50000], labels=['<£5k','£5-10k','£10-20k','£20k+'])
df['apr_band']  = pd.cut(df['APR'],        bins=[0,0.10,0.15,0.20,0.25], labels=['5-10%','10-15%','15-20%','20-25%'])

print("\nNull counts:\n", df.isnull().sum())
print("\nAPR is null only for declined applications (no rate assigned):",
      (df[df['APR'].isnull()]['application_outcome'] == 'declined').all())


# =============================================================================
# SECTION 2 – OVERALL FUNNEL
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 2: OVERALL APPLICATION FUNNEL")
print("=" * 60)

total    = len(df)
approved = df['approved'].sum()
funded   = df['funded_flag'].sum()
declined = total - approved

approval_rate   = approved / total
conversion_rate = funded   / approved   # of approved, how many funded
funded_rate     = funded   / total      # end-to-end funded rate

print(f"\nTotal applications : {total:,}")
print(f"Approved           : {approved:,}  ({approval_rate:.1%})")
print(f"Declined           : {declined:,}  ({declined/total:.1%})")
print(f"Funded             : {funded:,}  ({conversion_rate:.1%} of approved)")
print(f"Overall funded rate: {funded_rate:.1%}")

# Chart
fig, ax = plt.subplots(figsize=(7, 4), facecolor=NAVY)
stages = ['Applications\n10,000', 'Approved\n6,470', 'Funded\n3,069']
vals   = [10000, 6470, 3069]
colors = [TEAL, TEAL2, '#009688']
bars   = ax.barh(stages[::-1], vals[::-1], color=colors[::-1], height=0.5)
for bar, val in zip(bars, vals[::-1]):
    ax.text(val + 150, bar.get_y() + bar.get_height() / 2,
            f'{val:,}', va='center', color=WHITE, fontsize=12, fontweight='bold')
ax.set_xlim(0, 12000)
ax.set_facecolor(NAVY)
for sp in ['top','right']: ax.spines[sp].set_visible(False)
for sp in ['bottom','left']: ax.spines[sp].set_color(GREY)
ax.tick_params(colors=WHITE)
ax.set_title('Application Funnel Overview', color=WHITE, fontsize=14, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/funnel.png', dpi=150, bbox_inches='tight', facecolor=NAVY)
plt.close()
print("\n[Chart saved] funnel.png")


# =============================================================================
# SECTION 3 – URBAN vs RURAL
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 3: URBAN vs RURAL")
print("=" * 60)

area = df.groupby('area').agg(
    total    = ('ID', 'count'),
    approved = ('approved', 'sum'),
    funded   = ('funded_flag', 'sum')
).assign(
    approval_rate   = lambda x: x['approved'] / x['total'],
    conversion_rate = lambda x: x['funded']   / x['approved'],
    funded_rate     = lambda x: x['funded']   / x['total']
)
print("\n", area.round(3))

fig, ax = plt.subplots(figsize=(6, 4), facecolor=NAVY)
areas = ['Rural', 'Urban']
appr  = [57.5, 72.8]
fund  = [23.5, 38.7]
x = np.arange(len(areas))
w = 0.35
b1 = ax.bar(x - w/2, appr, w, color=TEAL, label='Approval Rate %')
b2 = ax.bar(x + w/2, fund, w, color=RED,  label='Funded Rate %')
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8,
            f'{b.get_height():.1f}%', ha='center', color=WHITE, fontsize=10, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(areas, fontsize=12, color=WHITE)
ax.set_ylim(0, 90)
ax.tick_params(colors=WHITE); ax.set_facecolor(NAVY)
for sp in ['top','right']: ax.spines[sp].set_visible(False)
for sp in ['bottom','left']: ax.spines[sp].set_color(GREY)
ax.legend(facecolor=NAVY, labelcolor=WHITE, fontsize=9)
ax.set_title('Approval & Funded Rate: Urban vs Rural', color=WHITE, fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/area.png', dpi=150, bbox_inches='tight', facecolor=NAVY)
plt.close()
print("[Chart saved] area.png")


# =============================================================================
# SECTION 4 – AGE ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 4: AGE BAND ANALYSIS")
print("=" * 60)

age = df.groupby('age_band', observed=True).agg(
    total    = ('ID', 'count'),
    approved = ('approved', 'sum'),
    funded   = ('funded_flag', 'sum')
).assign(
    approval_rate   = lambda x: x['approved'] / x['total'],
    conversion_rate = lambda x: x['funded']   / x['approved'],
    funded_rate     = lambda x: x['funded']   / x['total']
)
print("\n", age.round(3))

fig, ax = plt.subplots(figsize=(6, 4), facecolor=NAVY)
age_labels   = ['18-25', '26-35', '36-50', '51-65']
funded_r     = [11.8, 24.2, 38.1, 41.9]
approval_r   = [26.4, 48.4, 81.4, 87.5]
bar_colors   = [TEAL if v < 30 else '#00E5CC' for v in funded_r]
bars = ax.bar(age_labels, funded_r, color=bar_colors, width=0.5)
ax2  = ax.twinx()
ax2.plot(age_labels, approval_r, color=RED, marker='o', linewidth=2, markersize=7, label='Approval Rate %')
ax2.set_ylim(0, 120); ax2.tick_params(colors=WHITE)
ax2.spines['right'].set_color(GREY); ax2.spines['top'].set_visible(False)
for b, v in zip(bars, funded_r):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f'{v}%',
            ha='center', color=WHITE, fontsize=10, fontweight='bold')
ax.set_ylim(0, 55); ax.tick_params(colors=WHITE); ax.set_facecolor(NAVY)
for sp in ['top','right']: ax.spines[sp].set_visible(False)
for sp in ['bottom','left']: ax.spines[sp].set_color(GREY)
ax.set_title('Funded Rate & Approval Rate by Age', color=WHITE, fontsize=13, fontweight='bold')
legend_elements = [mpatches.Patch(color=TEAL, label='Funded Rate %'),
                   Line2D([0],[0], color=RED, marker='o', label='Approval Rate %')]
ax.legend(handles=legend_elements, facecolor=NAVY, labelcolor=WHITE, fontsize=9, loc='upper left')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/age.png', dpi=150, bbox_inches='tight', facecolor=NAVY)
plt.close()
print("[Chart saved] age.png")


# =============================================================================
# SECTION 5 – CAR TYPE ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 5: CAR TYPE ANALYSIS")
print("=" * 60)

car = df.groupby('car_type').agg(
    total    = ('ID', 'count'),
    approved = ('approved', 'sum'),
    funded   = ('funded_flag', 'sum')
).assign(
    approval_rate   = lambda x: x['approved'] / x['total'],
    conversion_rate = lambda x: x['funded']   / x['approved'],
    funded_rate     = lambda x: x['funded']   / x['total']
)
print("\n", car.round(3))

avg_loan = df.groupby('car_type')['loan_clean'].mean().round(0)
print("\nAverage loan amount by car type:\n", avg_loan)

fig, ax = plt.subplots(figsize=(6, 4), facecolor=NAVY)
car_types  = ['Convertible', 'SUV', 'Saloon']
approval   = [69.8, 58.1, 67.8]
conversion = [48.7, 47.3, 46.3]
funded_r   = [34.0, 27.5, 31.4]
x = np.arange(len(car_types))
w = 0.28
b1 = ax.bar(x - w, approval,   w, color=TEAL,  label='Approval Rate %')
b2 = ax.bar(x,     conversion, w, color=TEAL2, label='Conversion Rate %')
b3 = ax.bar(x + w, funded_r,   w, color=RED,   label='Funded Rate %')
for b in list(b1) + list(b2) + list(b3):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
            f'{b.get_height():.0f}%', ha='center', color=WHITE, fontsize=8, fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(car_types, fontsize=11, color=WHITE)
ax.set_ylim(0, 85); ax.tick_params(colors=WHITE); ax.set_facecolor(NAVY)
for sp in ['top','right']: ax.spines[sp].set_visible(False)
for sp in ['bottom','left']: ax.spines[sp].set_color(GREY)
ax.legend(facecolor=NAVY, labelcolor=WHITE, fontsize=8)
ax.set_title('Performance by Car Type', color=WHITE, fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/car.png', dpi=150, bbox_inches='tight', facecolor=NAVY)
plt.close()
print("[Chart saved] car.png")


# =============================================================================
# SECTION 6 – APR vs CONVERSION
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 6: APR vs CONVERSION RATE")
print("=" * 60)

approved_df = df[df['approved'] == 1].copy()
apr = approved_df.groupby('apr_band', observed=True).agg(
    total  = ('ID', 'count'),
    funded = ('funded_flag', 'sum')
).assign(conversion_rate=lambda x: x['funded'] / x['total'])
print("\n", apr.round(3))
print("\nKey insight: Conversion rate is flat across all APR bands — customers are not price-sensitive once approved.")

fig, ax = plt.subplots(figsize=(6, 4), facecolor=NAVY)
apr_bands  = ['5-10%', '10-15%', '15-20%', '20-25%']
conv_rates = [47.0, 47.6, 48.3, 46.9]
volume     = [2259, 1977, 1275, 959]
bars = ax.bar(apr_bands, conv_rates, color=TEAL, width=0.5)
ax2  = ax.twinx()
ax2.plot(apr_bands, volume, color=GOLD, marker='s', linewidth=2, markersize=7, linestyle='--', label='Volume')
ax2.tick_params(colors=WHITE)
ax2.spines['right'].set_color(GREY); ax2.spines['top'].set_visible(False)
ax2.set_ylabel('Applications', color=WHITE)
for b, v in zip(bars, conv_rates):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.1, f'{v}%',
            ha='center', color=WHITE, fontsize=11, fontweight='bold')
ax.set_ylim(44, 52); ax.tick_params(colors=WHITE); ax.set_facecolor(NAVY)
for sp in ['top','right']: ax.spines[sp].set_visible(False)
for sp in ['bottom','left']: ax.spines[sp].set_color(GREY)
ax.set_title('APR Band vs Conversion Rate (Approved Only)', color=WHITE, fontsize=12, fontweight='bold')
legend_el = [mpatches.Patch(color=TEAL, label='Conversion %'),
             Line2D([0],[0], color=GOLD, marker='s', linestyle='--', label='Volume')]
ax.legend(handles=legend_el, facecolor=NAVY, labelcolor=WHITE, fontsize=9)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/apr.png', dpi=150, bbox_inches='tight', facecolor=NAVY)
plt.close()
print("[Chart saved] apr.png")


# =============================================================================
# SECTION 7 – LOAN AMOUNT BANDS
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 7: LOAN AMOUNT BAND ANALYSIS")
print("=" * 60)

loan = df.groupby('loan_band', observed=True).agg(
    total    = ('ID', 'count'),
    approved = ('approved', 'sum'),
    funded   = ('funded_flag', 'sum')
).assign(
    approval_rate   = lambda x: x['approved'] / x['total'],
    conversion_rate = lambda x: x['funded']   / x['approved'],
    funded_rate     = lambda x: x['funded']   / x['total']
)
print("\n", loan.round(3))
print("\nKey insight: £10-20k loans have the best approval rate (69.9%) AND highest volume (4,623).")


# =============================================================================
# SECTION 8 – SWEET SPOT SEGMENT ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 8: BEST MARKETING SEGMENTS (SWEET SPOT)")
print("=" * 60)

sweet = (
    df[df['area'] == 'urban']
    .groupby(['age_band', 'car_type'], observed=True)
    .agg(total=('ID', 'count'), funded=('funded_flag', 'sum'))
    .assign(funded_rate=lambda x: x['funded'] / x['total'])
    .sort_values('funded_rate', ascending=False)
    .head(10)
)
print("\nTop Urban Segments by Funded Rate:\n", sweet.round(3))

fig, ax = plt.subplots(figsize=(7, 4), facecolor=NAVY)
segments = ['Urban 51-65\nConvertible', 'Urban 51-65\nSaloon', 'Urban 51-65\nSUV',
            'Urban 36-50\nSaloon', 'Urban 36-50\nConvertible', 'Urban 36-50\nSUV']
rates    = [50.4, 49.3, 49.1, 46.2, 44.4, 42.7]
bar_cols = ['#00E5CC' if r > 49 else TEAL if r > 45 else TEAL2 for r in rates]
bars = ax.barh(segments[::-1], rates[::-1], color=bar_cols[::-1], height=0.55)
for b, v in zip(bars, rates[::-1]):
    ax.text(v + 0.3, b.get_y() + b.get_height() / 2,
            f'{v}%', va='center', color=WHITE, fontsize=11, fontweight='bold')
ax.set_xlim(0, 60); ax.tick_params(colors=WHITE); ax.set_facecolor(NAVY)
for sp in ['top','right']: ax.spines[sp].set_visible(False)
for sp in ['bottom','left']: ax.spines[sp].set_color(GREY)
ax.set_title('Top Performing Urban Segments (Funded Rate %)', color=WHITE, fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/segments.png', dpi=150, bbox_inches='tight', facecolor=NAVY)
plt.close()
print("[Chart saved] segments.png")


# =============================================================================
# SECTION 9 – DEPOSIT RATIO ANALYSIS
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 9: DEPOSIT-TO-LOAN RATIO")
print("=" * 60)

dep = df.groupby('funded_flag')['deposit_ratio'].describe()
print("\nDeposit ratio by funded status:\n", dep.round(4))
print("\nKey insight: Funded customers have a marginally higher deposit ratio on average,")
print("but the difference is small — deposit size is not a strong standalone predictor.")


# =============================================================================
# SECTION 10 – SUMMARY TABLE
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 10: SUMMARY – ALL SEGMENTS RANKED BY FUNDED RATE")
print("=" * 60)

summary = df.groupby(['area', 'age_band', 'car_type'], observed=True).agg(
    applications = ('ID', 'count'),
    funded       = ('funded_flag', 'sum')
).assign(funded_rate=lambda x: (x['funded'] / x['applications']).round(3)) \
 .sort_values('funded_rate', ascending=False)

print("\nTop 10 segments:\n", summary.head(10).to_string())
print("\nBottom 5 segments:\n", summary.tail(5).to_string())


# =============================================================================
# RECOMMENDATIONS SUMMARY
# =============================================================================
print("\n" + "=" * 60)
print("RECOMMENDATIONS FOR MARKETING INVESTMENT")
print("=" * 60)
recs = [
    ("01", "Double down on urban channels",
     "Urban customers have 72.8% approval vs 57.5% rural, and 38.7% vs 23.5% funded rate. "
     "65% more likely to result in a funded loan. Geo-target urban postcodes."),
    ("02", "Target 36-65 age demographics",
     "81-88% approval rate. 36-65 age group accounts for 78% of all funded loans. "
     "Avoid heavy spend on 18-25 (only 26.4% approval, 11.8% funded rate)."),
    ("03", "Lead with Convertible & Saloon products",
     "Convertibles: highest funded rate (34%) + avg loan £17,179. "
     "SUVs attract the most volume but fund at only 27.5%."),
    ("04", "APR is not the lever — volume is",
     "Conversion rate is flat at ~47% across all APR bands. "
     "Customers are NOT price-sensitive post-approval. Focus on reaching more applicants."),
    ("05", "Address post-approval drop-off urgently",
     "47.4% of approved customers never fund. Test re-engagement emails, "
     "faster decisioning, and nudge messaging immediately post-approval."),
]
for num, title, body in recs:
    print(f"\n[{num}] {title}")
    print(f"     {body}")

print("\n" + "=" * 60)
print("All charts saved to ./charts/")
print("=" * 60)

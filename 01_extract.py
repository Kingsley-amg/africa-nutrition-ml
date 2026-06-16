"""
Step 1 - Build the Africa nutrition panel from the World Bank API.
Target: prevalence of undernourishment (% of population).
Predictors: socio-economic, agricultural, health and demographic determinants.
Country-year panel for African countries, 2000-2024.  No API key needed.

Run:  python 01_extract.py   ->  data/africa_nutrition.csv
"""
import time
from functools import reduce
from pathlib import Path
import requests
import pandas as pd

BASE = "https://api.worldbank.org/v2"
DATA = Path("data"); DATA.mkdir(exist_ok=True)
YEARS = "2000:2024"

TARGET = ("SN.ITK.DEFC.ZS", "undernourishment")   # % of population (the outcome)

PREDICTORS = {
    "NY.GDP.PCAP.PP.KD": "gdp_per_capita",          # PPP, constant intl $
    "SH.H2O.BASW.ZS":    "water_access",            # % basic drinking water
    "SH.STA.BASS.ZS":    "sanitation_access",       # % basic sanitation
    "SP.DYN.TFRT.IN":    "fertility_rate",          # births per woman
    "SP.RUR.TOTL.ZS":    "rural_pop_pct",           # % living rurally
    "AG.PRD.FOOD.XD":    "food_production_idx",      # food production index
    "SL.AGR.EMPL.ZS":    "agri_employment_pct",     # % employed in agriculture
    "NV.AGR.TOTL.ZS":    "agri_value_added_pct",    # agriculture, % of GDP
    "SP.POP.GROW":       "pop_growth",              # annual population growth %
    "SH.XPD.CHEX.PC.CD": "health_exp_per_capita",   # current US$
    "SE.PRM.ENRR":       "primary_enrollment",      # gross primary enrolment %
    "FP.CPI.TOTL.ZG":    "inflation",               # consumer price inflation %
}

def fetch(code, name):
    url = f"{BASE}/country/all/indicator/{code}"
    for _ in range(3):
        r = requests.get(url, params={"format": "json", "date": YEARS, "per_page": 20000}, timeout=60)
        j = r.json()
        if r.ok and isinstance(j, list) and len(j) > 1 and j[1]:
            df = pd.DataFrame([{"iso3": d["countryiso3code"], "year": int(d["date"]),
                                name: d["value"]} for d in j[1]])
            df = df[df["iso3"].str.len() == 3].dropna(subset=[name])
            return df.drop_duplicates(["iso3", "year"])
        time.sleep(2)
    raise RuntimeError(f"failed {code}")

print("downloading target + predictors ...")
frames = [fetch(*TARGET)] + [fetch(c, n) for c, n in PREDICTORS.items()]
panel = reduce(lambda l, r: l.merge(r, on=["iso3", "year"], how="outer"), frames)

# Country metadata; keep African countries only
meta = requests.get(f"{BASE}/country", params={"format": "json", "per_page": 400}, timeout=60).json()[1]
md = pd.DataFrame([{"iso3": c["id"], "country": c["name"].strip(),
                    "region": c["region"]["value"].strip(),
                    "income_group": c["incomeLevel"]["value"].strip()} for c in meta])
african = md[md.region.isin(["Sub-Saharan Africa",
                             "Middle East, North Africa, Afghanistan & Pakistan"])].copy()
# restrict the MENA bucket to actual African states
north_africa = {"DZA", "EGY", "LBY", "MAR", "TUN", "SDN", "ESH"}
african = african[(african.region == "Sub-Saharan Africa") | (african.iso3.isin(north_africa))]

panel = panel.merge(african, on="iso3", how="inner").sort_values(["country", "year"])
panel = panel.dropna(subset=["undernourishment"])          # need the target

panel.to_csv(DATA / "africa_nutrition.csv", index=False)
print("\n=== EXTRACT COMPLETE ===")
print(f"Countries: {panel.iso3.nunique()} | years: {panel.year.min()}-{panel.year.max()}")
print(f"Rows (with target): {len(panel):,}")
print("Predictor coverage (% non-missing):")
for n in PREDICTORS.values():
    print(f"  {n:22} {panel[n].notna().mean():.0%}")

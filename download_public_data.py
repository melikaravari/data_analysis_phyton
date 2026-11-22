"""
Downloads the small public data sets used by projects 16-18 and 20-29 (the Senate, recidivism and speech data come with Python packages) into ./data (run once; needs an internet connection).
Sources: the Rdatasets collection (Vincent Arel-Bundock), which mirrors data from R packages:
  - datasets::Puromycin  (Treloar 1974)           -> data/puromycin.csv
  - psych::bfi           (Revelle, psych package) -> data/bfi.csv
  - carData::GSSvocab    (NORC General Social Survey, via the carData package) -> data/gss_vocab.csv
Please cite the original sources in your own write-ups.
"""
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/"
FILES = {"datasets/Puromycin.csv": "puromycin.csv", "psych/bfi.csv": "bfi.csv", "carData/GSSvocab.csv": "gss_vocab.csv",
         "wooldridge/kielmc.csv": "kielmc.csv", "wooldridge/card.csv": "card.csv", "AER/CigarettesSW.csv": "cigarettes_sw.csv",
         "lme4/sleepstudy.csv": "sleepstudy.csv", "datasets/npk.csv": "npk.csv"}
OTHER = {  # (full URL, local name)
    "https://raw.githubusercontent.com/datasets/co2-ppm/main/data/co2-mm-mlo.csv": "co2_mauna_loa_monthly.csv",
    "https://raw.githubusercontent.com/datasets/global-temp/main/data/monthly.csv": "gistemp_monthly.csv",
    "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json": "us_states.geojson",
}
out = Path(__file__).parent / "data"
out.mkdir(exist_ok=True)
for remote, local in FILES.items():
    urllib.request.urlretrieve(BASE + remote, out / local)
    print("downloaded", local)
for url, local in OTHER.items():
    urllib.request.urlretrieve(url, out / local)
    print("downloaded", local)

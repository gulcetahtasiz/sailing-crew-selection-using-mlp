import numpy as np
import pandas as pd

np.random.seed(42)

N = 150
data = {}

# --- demographics & continuous features ---
data["sex"]        = np.random.choice(["M","F"], size=N, p=[0.6, 0.4])
data["age"]        = np.random.randint(18, 40, size=N)
data["height_cm"]  = np.random.normal(175, 10, size=N).astype(int)
data["weight_kg"]  = np.random.normal(70, 12, size=N).astype(int)

data["upper_watts"] = np.random.normal(500, 100, size=N).clip(200, 800).astype(int)
data["grip_kg"]     = np.random.normal(45, 8, size=N).clip(20, 70).astype(int)
data["plank_sec"]   = np.random.normal(120, 40, size=N).clip(30, 300).astype(int)
data["agility_ms"]  = np.random.normal(800, 150, size=N).clip(500, 1500).astype(int)

data["decision_ms"] = np.random.normal(350, 60, size=N).clip(200, 600).astype(int)
data["attention"]   = np.random.randint(50, 101, size=N)
data["years_exp"]   = np.random.poisson(4, size=N).clip(0, 15)

# --- extra features expected by training code ---
data["exp_dinghy"] = np.random.randint(0, 2, size=N)          # 0/1
data["exp_keel"]   = np.random.randint(0, 2, size=N)          # 0/1
data["starts"]     = np.random.poisson(10, size=N)            # race starts
data["penalties"]  = np.random.poisson(1, size=N)             # penalties

df = pd.DataFrame(data)

# --- name pools (Turkish) ---
female_first_names = [
    "Ayşe","Fatma","Elif","Zeynep","Merve","Esra","Gözde","Selin","Seda","Buse",
    "İrem","Sude","Sena","Ece","Naz","Ceren","Yasemin","Derya","Aylin","Hülya",
    "Melisa","Tuğçe","Dilara","Gül","Nisanur","Ceyda","Pelin","İlayda","Damla","Hazal"
]
male_first_names = [
    "Ahmet","Mehmet","Mustafa","Ali","Osman","Emre","Can","Mert","Burak","Oğuz",
    "Berk","Uğur","Hakan","Onur","Kaan","Eren","Yusuf","Ömer","Furkan","Kerem",
    "Deniz","Tolga","Serkan","Tunç","Barış","Cem","Volkan","Sinan","Arda","Yiğit"
]
last_names = [
    "Yılmaz","Şahin","Demir","Çelik","Yıldız","Yıldırım","Aydın","Öztürk","Arslan","Doğan",
    "Kılıç","Aslan","Kaya","Koç","Kurt","Aksoy","Polat","Kara","Duran","Tekin",
    "Erden","Avcı","Keskin","Bozkurt","Ekinci","Ersoy","Bulut","Güneş","Taş","Uzun"
]

# --- generate unique full names and assign as athlete_id ---
name_counts = {}
def pick_full_name(sex: str) -> str:
    first = np.random.choice(female_first_names if sex == "F" else male_first_names)
    last  = np.random.choice(last_names)
    base  = f"{first} {last}"
    cnt   = name_counts.get(base, 0) + 1
    name_counts[base] = cnt
    return base if cnt == 1 else f"{base} #{cnt}"

df["athlete_id"] = [pick_full_name(s) for s in df["sex"]]

# --- label rules ---
labels = {}
labels["basustu"]   = ((df["agility_ms"] < 900) & (df["plank_sec"] > 100)).astype(int)
labels["direkdibi"] = ((df["upper_watts"] > 550) & (df["grip_kg"] > 40)).astype(int)
labels["piyano"]    = ((df["attention"] > 70) & (df["decision_ms"] < 400)).astype(int)
labels["trim1"]     = ((df["attention"] > 65) & (df["years_exp"] > 3)).astype(int)
labels["trim2"]     = ((df["years_exp"] > 3) & (df["agility_ms"] < 1000)).astype(int)
labels["anayelken"] = ((df["upper_watts"] > 450) & (df["years_exp"] > 2)).astype(int)
labels["dumen"]     = ((df["decision_ms"] < 370) & (df["attention"] > 60) & (df["years_exp"] > 4)).astype(int)

for k, v in labels.items():
    df[k] = v

# --- put athlete_id first (optional) ---
cols = ["athlete_id"] + [c for c in df.columns if c != "athlete_id"]
df = df[cols]

# --- save ---
out_path = "sailing_synthetic_groundtruth_150_named.csv"
df.to_csv(out_path, index=False)  # Excel için istersen encoding="utf-8-sig" ekleyebilirsin
print(f"Saved {out_path} with shape:", df.shape)

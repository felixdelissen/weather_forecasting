import numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ------------------------------------------------------------------
# 0. Data -> pivot (index horaire complet, NaN = heures manquantes)
# ------------------------------------------------------------------
df = pd.read_csv("temperature.csv")
df.columns = ["station", "ts", "temp"]
df["ts"] = pd.to_datetime(df["ts"], format="%m/%d/%Y %I:%M:%S %p")
pivot = (df.pivot_table(index="ts", columns="station", values="temp", aggfunc="mean")
           .resample("1h").mean())

# ------------------------------------------------------------------
# 1. Features par station
#    <<< MASK ICI : obs est calcule AVANT interpolation et voyage
#        avec les features dans la colonne "ok" >>>
#    <<< DUMMY ICI (1/2) : on tamponne chaque bloc avec le nom
#        de sa station dans la colonne "station" >>>
# ------------------------------------------------------------------
def make_features(s, name):
    obs = s.notna()                                   # photo du reel AVANT fill
    s = s.interpolate(limit=3, limit_area="inside")   # fill leger, features only

    X = pd.DataFrame(index=s.index)
    for l in (1, 2):
        X[f"lag_{l}"] = s.shift(l)
    X["dtemp"]    = s.shift(1) - s.shift(2)
    X["rmean_24"] = s.shift(1).rolling(24).mean()     # garde: sert de baseline
    h, d = s.index.hour, s.index.dayofyear
    X["hour_sin"] = np.sin(2*np.pi*h/24);    X["hour_cos"] = np.cos(2*np.pi*h/24)
    X["doy_sin"]  = np.sin(2*np.pi*d/365.25); X["doy_cos"] = np.cos(2*np.pi*d/365.25)

    X["y"]       = s
    X["ok"]      = obs                                 # <- mask
    X["station"] = name                                # <- dummy (1/2)
    return X

data = pd.concat([make_features(pivot[c], c) for c in pivot.columns]).sort_index()

# ------------------------------------------------------------------
# 2. DUMMY (2/2) : la colonne station devient des 0/1
#    drop_first evite la colinearite avec l'intercept
# ------------------------------------------------------------------
data = pd.get_dummies(data, columns=["station"], drop_first=True, dtype=float)

# ------------------------------------------------------------------
# 3. <<< MASK APPLIQUE ICI : on vire (a) les lignes aux features
#    manquantes, (b) les lignes dont la CIBLE n'est pas une vraie
#    mesure. Les valeurs interpolees restent dans les FEATURES. >>>
# ------------------------------------------------------------------
data = data.dropna()              # (a)
data = data[data["ok"]]           # (b) jamais de cible fabriquee

X = data.drop(columns=["y", "ok"])
y = data["y"]

# ------------------------------------------------------------------
# 4. Test reserve AVANT tout
# ------------------------------------------------------------------
cut = int(len(X) * 0.7)
X_train, X_test = X.iloc[:cut], X.iloc[cut:]
y_train, y_test = y.iloc[:cut], y.iloc[cut:]

# ------------------------------------------------------------------
# 5. Grid search dans le train (fit 1) + refit auto = modele final
# ------------------------------------------------------------------
tscv = TimeSeriesSplit(n_splits=5)
grid = GridSearchCV(make_pipeline(StandardScaler(), Ridge()),
                    {"ridge__alpha": [0.01, 0.1, 1, 10, 100]},
                    cv=tscv, scoring="neg_mean_absolute_error")
grid.fit(X_train, y_train)
best_alpha = grid.best_params_["ridge__alpha"]

# ------------------------------------------------------------------
# 6. Diagnostic par fold (dans le train, alpha tune)
# ------------------------------------------------------------------
model = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha))
rows = []
for fold, (tr, te) in enumerate(tscv.split(X_train)):
    Xa, Xb = X_train.iloc[tr], X_train.iloc[te]
    ya, yb = y_train.iloc[tr], y_train.iloc[te]
    model.fit(Xa, ya)
    p = model.predict(Xb)
    rows.append({"fold": fold,
                 "pers_mae":  mean_absolute_error(yb, Xb["lag_1"]),
                 "roll_mae":  mean_absolute_error(yb, Xb["rmean_24"]),
                 "ridge_mae": mean_absolute_error(yb, p),
                 "ridge_rmse": np.sqrt(mean_squared_error(yb, p))})
print(pd.DataFrame(rows).set_index("fold").round(4))

# ------------------------------------------------------------------
# 7. Verdict final, une fois
# ------------------------------------------------------------------
print("\nalpha :", best_alpha)
print(f"TEST persistence : {mean_absolute_error(y_test, X_test['lag_1']):.4f}")
print(f"TEST ridge       : {mean_absolute_error(y_test, grid.predict(X_test)):.4f}")

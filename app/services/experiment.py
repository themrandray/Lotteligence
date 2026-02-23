import numpy as np
import pandas as pd

# Sklearn metrikas modeļu novērtēšanai: 
# log_loss: mēra prognožu ticamību (jo mazāks, jo labāk) 
# mean_squared_error: Brier score (jo mazāks, jo labāk)
from sklearn.metrics import log_loss, mean_squared_error

# Izslēdz brīdinājumus par matricas reizināšanu
import warnings 
warnings.filterwarnings("ignore", message=".*matmul.*", category=RuntimeWarning)

# Modeļu būvēšana un prognozēšana
from .models import build_logreg_sgd, build_random_forest, build_xgboost_like, fit_and_predict

def run_experiment(df_norm: pd.DataFrame, lottery: str, window: int = 1):
    # Izpilda eksperimentu ar trim modeļiem (LogReg, RandomForest, XGBoost-like)
    # Izmanto lagged features: prev_vec -> curr_vec
    # Atgriež metrikas un informāciju par treniņu/testu periodiem

    # Nosaka loterijas parametrus
    if lottery == "viking":
        max_num = 48
        k_main = 6
    elif lottery == "euro":
        max_num = 50
        k_main = 5
    else:
        raise ValueError("Nezināms loterijas tips eksperimentam")

    # Sagatavo lagged features (prev_vec un curr_vec)
    df_feat = _prepare_lagged_features(df_norm, max_num=max_num)
    df_feat = df_feat.sort_values("date").reset_index(drop=True)

    n = len(df_feat)
    if n < 10:
        raise ValueError("Nepietiek datu pēc lagged apstrādes (vajag vismaz 10 rindas)")

    # 70% treniņam, 30% testam
    split_idx = int(n * 0.7)
    train = df_feat.iloc[:split_idx]
    test = df_feat.iloc[split_idx:]

    # Sagatavo X un Y matricas
    X_train = np.stack(train["prev_vec"].values)
    Y_train = np.stack(train["curr_vec"].values)
    X_test = np.stack(test["prev_vec"].values)
    Y_test = np.stack(test["curr_vec"].values)

    # Datumu diapazoni (informatīvi)
    train_date_from = train["date"].min().date().isoformat()
    train_date_to = train["date"].max().date().isoformat()
    test_date_from = test["date"].min().date().isoformat()
    test_date_to = test["date"].max().date().isoformat()

    results = []

    # Modeļu saraksts
    models = [
        ("logreg_sgd", build_logreg_sgd()),
        ("random_forest", build_random_forest()),
        ("xgboost", build_xgboost_like()),
    ]

    # Izpilda katru modeli
    for name, model in models:
        proba = fit_and_predict(model, X_train, Y_train, X_test)

        # Aizsardzība pret 0 un 1 (logloss nevar aprēķināt)
        proba_clipped = np.clip(proba, 1e-6, 1 - 1e-6)

        # Metrikas
        ll = log_loss(Y_test.ravel(), proba_clipped.ravel())
        brier = mean_squared_error(Y_test.ravel(), proba_clipped.ravel())
        hit_k_main = _hit_at_k(Y_test, proba_clipped, k=k_main)
        hit_10 = _hit_at_k(Y_test, proba_clipped, k=10)

        # Rezultātu rinda
        res = {
            "model": name,
            "logloss": float(ll),
            "brier": float(brier),
            "hit_k_main": float(hit_k_main),
            "hit_10": float(hit_10),
            "k_main": int(k_main),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "train_date_from": train_date_from,
            "train_date_to": train_date_to,
            "test_date_from": test_date_from,
            "test_date_to": test_date_to,
            "window": int(window),
        }
        results.append(res)

    return results

def _prepare_lagged_features(df_norm: pd.DataFrame, max_num: int) -> pd.DataFrame:
    # Pārvērš katru izlozi par one-hot vektoru (garums = max_num)
    # Izveido prev_vec (iepriekšējā izloze) un curr_vec (pašreizējā izloze)

    rows = []
    for _, row in df_norm.iterrows():
        mains = [row["n1"], row["n2"], row["n3"], row["n4"], row["n5"]]
        if not pd.isna(row.get("n6", pd.NA)):
            mains.append(row["n6"])
        mains_clean = [int(x) for x in mains if not pd.isna(x)]

        vec = np.zeros(max_num, dtype=int)
        for m in mains_clean:
            if 1 <= m <= max_num:
                vec[m - 1] = 1

        rows.append({"date": pd.to_datetime(row["date"]), "vec": vec})

    df_vec = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    prev_vecs = [None] + df_vec["vec"].tolist()[:-1]
    df_vec["prev_vec"] = prev_vecs
    df_vec["curr_vec"] = df_vec["vec"]

    return df_vec.dropna(subset=["prev_vec"]).reset_index(drop=True)

def _hit_at_k(Y_true: np.ndarray, proba: np.ndarray, k: int) -> float:
    # Aprēķina hit@k — cik bieži patiesie skaitļi ir starp top-k prognozētajiem

    hits = []
    for y, p in zip(Y_true, proba):
        top_idx = np.argsort(p)[-k:]
        top_set = set(top_idx.tolist())
        true_idx = set(np.where(y == 1)[0].tolist())
        inter = len(true_idx.intersection(top_set))
        hits.append(inter / float(k))
    return float(np.mean(hits))
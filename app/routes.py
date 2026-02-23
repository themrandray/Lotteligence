# Flask komponentes:
# - Blueprint: ļauj sadalīt maršrutus pa moduļiem
# - render_template: ielādē HTML veidnes
# - request: nolasa formu datus un augšupielādētos failus
from flask import Blueprint, render_template, request

# Pathlib nodrošina ērtu un drošu darbu ar failu ceļiem (platformneatkarīgi)
from pathlib import Path

# datetime tiek izmantots laika zīmoga (timestamp) izveidei saglabātajiem failiem
from datetime import datetime

# Projekta servisi datu apstrādei:
# - read_table: nolasa CSV/XLSX failu pandas DataFrame formātā
# - normalize_any: normalizē datus un pārbauda loterijas tipu
from .services.dataset import read_table, normalize_any

# Eksperimentu izpilde (modeļi, prognozes utt.)
from .services.experiment import run_experiment

main_bp = Blueprint("main", __name__)

@main_bp.route("/", methods=["GET"])
def index():
    # Atgriež sākuma lapu ar noklusējuma iestatījumiem
    return render_template(
        "index.html",
        error=None,
        results=None,
        form_state={
            "lottery": "viking",
            "file_format": "raw",
            "window": "1",
        },
        status="idle",
    )

@main_bp.route("/run", methods=["POST"])
def run():
    # Apstrādā augšupielādēto failu un palaiž eksperimentu
    error = None
    results = None

    # Nolasa formā ievadītos parametrus
    lottery = request.form.get("lottery", "viking")
    file_format = request.form.get("file_format", "raw")
    window_str = request.form.get("window", "1")

    form_state = {
        "lottery": lottery,
        "file_format": file_format,
        "window": window_str,
    }

    # Validē loga parametru
    try:
        window = int(window_str)
        if window <= 0:
            raise ValueError
    except ValueError:
        error = "Loga parametrs ir jābūt pozitīvam veselam skaitlim"
        return render_template(
            "index.html",
            error=error,
            results=None,
            form_state=form_state,
            status="idle",
        )

    # Pārbauda, vai fails ir augšupielādēts
    file = request.files.get("dataset")
    if not file or file.filename == "":
        error = "Lūdzu augšupielādējiet datu failu"
        return render_template(
            "index.html",
            error=error,
            results=None,
            form_state=form_state,
            status="idle",
        )

    # Saglabā failu lokāli
    upload_dir = Path(main_bp.root_path).resolve().parent / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    safe_name = file.filename.replace(" ", "_")
    saved_path = upload_dir / safe_name
    file.save(saved_path)

    try:
        # Nolasa failu ar pandas
        df_raw = read_table(saved_path)

        # Normalizē datus un pārbauda loterijas tipu
        df_norm = normalize_any(df_raw, lottery=lottery, file_format=file_format)

        # Palaiž eksperimentu
        results = run_experiment(df_norm, lottery=lottery, window=window)

        # Saglabā rezultātus
        _save_outputs(df_norm, results, lottery, window)

        status = "done"

    except Exception as exc:
        error = str(exc)
        status = "idle"

    return render_template(
        "index.html",
        error=error,
        results=results,
        form_state=form_state,
        status=status,
    )

def _timestamp():
    # Izveido laika zīmogu vēstures ierakstiem (datums + laiks milisekundēs)
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S.") + f"{int(now.microsecond / 1000):03d}"

def _save_outputs(df_norm, results, lottery, window):
    # Saglabā tikai trīs failus:
    # - normalized_latest.csv (pēdējais normalizētais datasets)
    # - results_latest.csv (pēdējie eksperimenta rezultāti)
    # - results_history.csv (visu skrējienu vēsture)

    from flask import current_app
    import pandas as pd

    outputs_dir = current_app.config["OUTPUTS_DIR"]

    # 1) Saglabā pēdējo normalizēto datasetu
    df_norm.to_csv(outputs_dir / "normalized_latest.csv", index=False)

    # 2) Saglabā pēdējos rezultātus
    df_res = pd.DataFrame(results)
    df_res.to_csv(outputs_dir / "results_latest.csv", index=False)

    # 3) Pievieno rezultātus vēsturei
    history_path = outputs_dir / "results_history.csv"
    df_res_with_meta = df_res.copy()
    df_res_with_meta["timestamp"] = _timestamp()
    df_res_with_meta["lottery"] = lottery
    df_res_with_meta["window"] = window

    if history_path.exists():
        df_old = pd.read_csv(history_path)
        df_all = pd.concat([df_old, df_res_with_meta], ignore_index=True)
        df_all.to_csv(history_path, index=False)
    else:
        df_res_with_meta.to_csv(history_path, index=False)
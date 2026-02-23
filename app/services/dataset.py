import pandas as pd
from pathlib import Path

# Definēju kolonnas, kas raksturīgas LatLoto RAW formātam
# Tas palīdz saprast, ka fails ir RAW un pirms izmantošanas tas jāapstrādā īpašā veidā
RAW_COLUMNS = ["Izlozes Nr.", "Datums", "Izlozētie skaitļi"]


def read_table(path: Path) -> pd.DataFrame:
    # Nolasa CSV vai Excel failu, automātiski nosakot formātu pēc paplašinājuma
    # Šī funkcija ļauj lietotājam izmantot dažādus failu tipus bez papildus iestatījumiem

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    elif suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    else:
        raise ValueError("Neatbalstīts faila formāts. Izmanto CSV vai XLSX")


def is_latloto_raw(df: pd.DataFrame) -> bool:
    # Pārbauda, vai fails izskatās pēc LatLoto RAW formāta
    # Ja visas RAW_COLUMNS kolonnas ir klāt, varu droši turpināt apstrādi

    return all(col in df.columns for col in RAW_COLUMNS)


def is_prepared(df: pd.DataFrame) -> bool:
    # Pārbauda, vai fails jau ir sagatavots vienotajā formātā (n1, n2... u.c.)
    # Tas palīdz izvairīties no kļūdām, ja lietotājs izvēlas nepareizu faila tipu

    required = ["draw_no", "date", "n1", "n2", "n3", "n4", "n5"]
    return all(col in df.columns for col in required)


def _parse_numbers_list(text: str):
    # Pārvērš tekstu ar skaitļiem par sarakstu ar veseliem skaitļiem
    # Funkcija ir pielāgota dažādiem formātiem - komatiem, atstarpēm, semikoliem
    
    if pd.isna(text):
        return []
    if isinstance(text, (int, float)):
        return [int(text)]

    s = str(text)
    for sep in [",", ";"]:
        s = s.replace(sep, " ")

    parts = [p for p in s.split() if p.strip() != ""]
    nums = []
    for p in parts:
        try:
            nums.append(int(p))
        except ValueError:
            # Ja gadās kāds nederīgs simbols, to vienkārši ignorē
            continue
    return nums


def _parse_main_and_bonus(value):
    # Sadala skaitļu virkni galvenajos un bonusu skaitļos
    # Ja ir "+", tad pa kreisi ir galvenie skaitļi, pa labi — bonusi
    # Ja "+" nav, tad pieņemu, ka bonusu nav
    
    if pd.isna(value):
        return [], []

    text = str(value)
    if "+" in text:
        left, right = text.split("+", 1)
        mains = _parse_numbers_list(left)
        bonuses = _parse_numbers_list(right)
    else:
        mains = _parse_numbers_list(text)
        bonuses = []

    return mains, bonuses


def _detect_lottery_from_numbers(main_counts, bonus_counts, max_main, max_bonus):
    # Mēģina noteikt loterijas tipu pēc datu struktūras
    
    # Viking Lotto: 6 galvenie skaitļi (1..48) un 1 bonusa skaitlis (1..5)
    if max_main <= 48 and all(c == 6 for c in main_counts) and all(b == 1 for b in bonus_counts):
        return "viking"

    # Eurojackpot: 5 galvenie skaitļi + 2 bonusi, diapazoni 1..50 un 1..12
    if max_main <= 50 and all(c == 5 for c in main_counts) and all(b in (0, 2) for b in bonus_counts):
        return "euro"

    return "unknown"


def normalize_any(df_raw: pd.DataFrame, lottery: str, file_format: str) -> pd.DataFrame:
    # Apstrādā RAW un prepared failus un pārvērš tos vienotā formātā
    # Pēc normalizācijas pārbauda, lai nesajauktu Viking Lotto un Eurojackpot datus

    
    if file_format == "raw":
        if not is_latloto_raw(df_raw):
            raise ValueError("RAW formāts izvēlēts, bet trūkst nepieciešamās kolonnas")
        df_norm = _normalize_raw(df_raw)

    elif file_format == "prepared":
        if not is_prepared(df_raw):
            raise ValueError("Prepared formāts izvēlēts, bet trūkst nepieciešamās kolonnas")
        df_norm = _normalize_prepared(df_raw)

    else:
        raise ValueError("Nezināms file_format parametrs")

    # Šeit notiek galvenā drošības pārbaude
    _validate_lottery_safety(df_norm, lottery)

    return df_norm


def _normalize_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    # Normalizē LatLoto RAW formātu uz vienotu kolonnu struktūru
    # Tiek izveidotas kolonnas n1..n6 un b1..b2, lai visi dati būtu vienādā formā

    records = []

    for _, row in df_raw.iterrows():
        draw_no = row["Izlozes Nr."]
        date = row["Datums"]

        mains, bonuses = _parse_main_and_bonus(row["Izlozētie skaitļi"])
        mains_sorted = sorted(int(n) for n in mains)
        bonuses_sorted = sorted(int(n) for n in bonuses)

        rec = {
            "draw_no": draw_no,
            "date": pd.to_datetime(date, dayfirst=True),
        }

        # Aizpilda n1..n6 (ja ir mazāk skaitļu, trūkstošie paliek None)
        for i in range(5):
            rec[f"n{i+1}"] = mains_sorted[i] if i < len(mains_sorted) else None
        rec["n6"] = mains_sorted[5] if len(mains_sorted) > 5 else None

        # Aizpilda bonusu skaitļus
        rec["b1"] = bonuses_sorted[0] if len(bonuses_sorted) > 0 else None
        rec["b2"] = bonuses_sorted[1] if len(bonuses_sorted) > 1 else None

        records.append(rec)

    df_norm = pd.DataFrame(records)
    df_norm = df_norm.sort_values("date").reset_index(drop=True)
    return df_norm


def _normalize_prepared(df_raw: pd.DataFrame) -> pd.DataFrame:
    # Apstrādā jau sagatavotu failu, nodrošinot, ka visas nepieciešamās kolonnas pastāv
    # Ja kāda kolonna trūkst, tā tiek izveidota automātiski

    df = df_raw.copy()
    df["date"] = pd.to_datetime(df["date"], dayfirst=True)

    cols = ["draw_no", "date", "n1", "n2", "n3", "n4", "n5", "n6", "b1", "b2"]
    for c in cols:
        if c not in df.columns:
            df[c] = None

    # Pārvērš skaitļus par Int64 (ļauj saglabāt arī NaN)
    for c in ["n1", "n2", "n3", "n4", "n5", "n6", "b1", "b2"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    df = df[cols]
    df = df.sort_values("date").reset_index(drop=True)
    return df


def _validate_lottery_safety(df_norm: pd.DataFrame, lottery: str):
    # Šī ir drošības funkcija, kas pārbauda, vai lietotāja izvēlētais loterijas tips atbilst faktiskajiem datiem
    # Tas novērš situācijas, kur:
    # - Viking Lotto dati tiek analizēti kā Eurojackpot
    # - Eurojackpot dati tiek analizēti kā Viking Lotto
    # - dati satur skaitļus ārpus atļautā diapazona
    # Ja tiek konstatēta neatbilstība, funkcija izmet kļūdu un aptur apstrādi

    main_counts = []
    bonus_counts = []
    max_main = 0
    max_bonus = 0

    for _, row in df_norm.iterrows():
        mains = [row["n1"], row["n2"], row["n3"], row["n4"], row["n5"]]
        if not pd.isna(row.get("n6", pd.NA)):
            mains.append(row["n6"])
        mains_clean = [int(x) for x in mains if not pd.isna(x)]

        bonuses = []
        if "b1" in df_norm.columns and not pd.isna(row.get("b1", pd.NA)):
            bonuses.append(int(row["b1"]))
        if "b2" in df_norm.columns and not pd.isna(row.get("b2", pd.NA)):
            bonuses.append(int(row["b2"]))

        main_counts.append(len(mains_clean))
        bonus_counts.append(len(bonuses))

        if mains_clean:
            max_main = max(max_main, max(mains_clean))
        if bonuses:
            max_bonus = max(max_bonus, max(bonuses))

    detected = _detect_lottery_from_numbers(main_counts, bonus_counts, max_main, max_bonus)

    if lottery == "viking":
        if detected == "euro":
            raise ValueError("Fails izskatās pēc Eurojackpot, bet izvēlēts 'viking'")
        if max_main > 48:
            raise ValueError("Galvenie skaitļi pārsniedz 48 — tas nav derīgs Viking Lotto")

    elif lottery == "euro":
        if detected == "viking":
            raise ValueError("Fails izskatās pēc Viking Lotto, bet izvēlēts 'euro'")
        if max_main > 50:
            raise ValueError("Galvenie skaitļi pārsniedz 50 — tas nav derīgs Eurojackpot")
        if max_bonus > 12:
            raise ValueError("Bonusa skaitļi pārsniedz 12 — tas nav derīgs Eurojackpot")

    else:
        raise ValueError("Nezināms loterijas tips")
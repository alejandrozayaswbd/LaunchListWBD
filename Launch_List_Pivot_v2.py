import pandas as pd
from pathlib import Path
import os
import re
import numpy as np

from datetime import datetime

import warnings 


warnings.filterwarnings('ignore')
pd.set_option("display.max_column", None)

# ========================
# Funciones
# ========================

cols_map_secondary = {
    "A | P&S ADR Secondary Max Brand_x": "A | P&S ADR Secondary Max Brand_y",
    "A | P&S CYP Secondary Max Brand_x": "A | P&S CYP Secondary Max Brand_y",
    "A | P&S CIS Secondary Max Brand_x": "A | P&S CIS Secondary Max Brand_y",
    "A | P&S GAL Secondary Max Brand": "A | P&S GAU Secondary Max Brand",
    "A | P&S GRE Secondary Max Brand_x": "A | P&S GRE Secondary Max Brand_y",
    "A | P&S ICE Secondary Max Brand_x": "A | P&S ICE Secondary Max Brand_y",
    "A | P&S IRE Secondary Max Brand_x": "A | P&S IRE Secondary Max Brand_y",
    "A | P&S ISR Secondary Max Brand_x": "A | P&S ISR Secondary Max Brand_y",
    "A | P&S ITA Secondary Max Brand_x": "A | P&S ITA Secondary Max Brand_y",
    "A | P&S MAL Secondary Max Brand_x": "A | P&S MAL Secondary Max Brand_y",
    "A | P&S UKR Secondary Max Brand_x": "A | P&S UKR Secondary Max Brand_y",
    "A | P&S UK Secondary Max Brand_x": "A | P&S UK Secondary Max Brand_y",
    "A | P&S LUX Secondary Max Brand_x": "A | P&S LUX Secondary Max Brand_y",
    "A | P&S SWI Secondary Max Brand": "A | P&S SWL Secondary Max Brand",
    "A | P&S BAL Secondary Max Brand copy":"A | P&S BAL Secondary Max Brand"
    
}

cols_map_primary = {
    "A | P&S ADR Primary Max Brand_x": "A | P&S ADR Primary Max Brand_y",
    "A | P&S CYP Primary Max Brand_x": "A | P&S CYP Primary Max Brand_y",
    "A | P&S CIS Primary Max Brand_x": "A | P&S CIS Primary Max Brand_y",
    "A | P&S GAL Primary Max Brand": "A | P&S GAU Primary Max Brand",
    "A | P&S GRE Primary Max Brand_x": "A | P&S GRE Primary Max Brand_y",
    "A | P&S ICE Primary Max Brand_x": "A | P&S ICE Primary Max Brand_y",
    "A | P&S IRE Primary Max Brand_x": "A | P&S IRE Primary Max Brand_y",
    "A | P&S ISR Primary Max Brand_x": "A | P&S ISR Primary Max Brand_y",
    "A | P&S ITA Primary Max Brand_x": "A | P&S ITA Primary Max Brand_y",
    "A | P&S MAL Primary Max Brand_x": "A | P&S MAL Primary Max Brand_y",
    "A | P&S UKR Primary Max Brand_x": "A | P&S UKR Primary Max Brand_y",
    "A | P&S UK Primary Max Brand_x": "A | P&S UK Primary Max Brand_y",
    "A | P&S LUX Primary Max Brand_x": "A | P&S LUX Primary Max Brand_y",
    "A | P&S SWL Primary Max Brand_x": "A | P&S SWL Primary Max Brand_y",
    "A | P&S BAL Primary Max Brand_x": "A | P&S BAL Primary Max Brand_y"
}

cols_map_end_date = {
    "A | P&S ADR End Date_x":"A | P&S ADR End Date_y",
    "A | P&S CYP End Date_x":"A | P&S CYP End Date_y",
    "A | P&S CIS End Date_x":"A | P&S CIS End Date_y",
    "A | P&S GAL End Date_x":"A | P&S GAL End Date_y",
    "A | P&S GRE End Date_x":"A | P&S GRE End Date_y",
    "A | P&S ICE End Date_x":"A | P&S ICE End Date_y",
    "A | P&S IRE End Date_x":"A | P&S IRE End Date_y",
    "A | P&S ISR End Date_x":"A | P&S ISR End Date_y",
    "A | P&S Italy End Date":"A | P&S ITA End Date",
    "A | P&S MAL End Date_x":"A | P&S MAL End Date_y",
    "A | P&S UKR End Date_x":"A | P&S UKR End Date_y",
    "A | P&S UK End Date_x":"A | P&S UK End Date_y",
    "A | P&S LUX End Date_x":"A | P&S LUX End Date_y",
    "A | P&S SWL End Date_x":"A | P&S SWL End Date_y",
    "A | P&S GAU End Date_x":"A | P&S GAU End Date_y",
    "A | P&S BAL End Date_x":"A | P&S BAL End Date_y"
}

cols_map_premiere_date = {
    "A | P&S ADR Planned Premiere Date_x":"A | P&S ADR Planned Premiere Date_y",
    "A | P&S CYP Planned Premiere Date_x":"A | P&S CYP Planned Premiere Date_y",
    "A | P&S CIS Planned Premiere Date_x":"A | P&S CIS Planned Premiere Date_y",
    "A | P&S GAU Planned Premiere Date_x":"A | P&S GAU Planned Premiere Date_y",
    "A | P&S GRE Planned Premiere Date_x":"A | P&S GRE Planned Premiere Date_y",
    "A | P&S ICE Planned Premiere Date_x":"A | P&S ICE Planned Premiere Date_y",
    "A | P&S IRE Planned Premiere Date_x":"A | P&S IRE Planned Premiere Date_y",
    "A | P&S ISR Planned Premiere Date_x":"A | P&S ISR Planned Premiere Date_y",
    "A | P&S ITA Planned Premiere Date_x":"A | P&S ITA Planned Premiere Date_y",
    "A | P&S MAL Planned Premiere Date_x":"A | P&S MAL Planned Premiere Date_y",
    "A | P&S UKR Planned Premiere Date_x":"A | P&S UKR Planned Premiere Date_y",
    "A | P&S UK Planned Premiere Date_x":"A | P&S UK Planned Premiere Date_y",
    "A | P&S LUX Planned Premiere Date_x":"A | P&S LUX Planned Premiere Date_y",
    "A | P&S SWL Planned Premiere Date_x":"A | P&S SWL Planned Premiere Date_y",
    "A | P&S GAU Planned Premiere Date_x":"A | P&S GAU Planned Premiere Date_y",
    "A | P&S BAL Planned Premiere Date_x":"A | P&S BAL Planned Premiere Date_y"
}


today = datetime.today().strftime("%b%d").upper()

def split_region_string(s: str):
    """
    Convierte una cadena gigante de regiones + descripciones
    en una lista donde cada elemento corresponde a una región.
    
    Args:
        s (str): Cadena de entrada
    
    Returns:
        list[str]: Lista de strings separados por región
    """
    # Paso 1: separar por comas
    parts = [p.strip() for p in s.split(",")]

    result = []
    buffer = []

    # Paso 2: identificar el inicio de un bloque (ej. "ADR - ...")
    for p in parts:
        if re.match(r"^[A-Z]{2,3}\s-\s", p):  # detecta inicio de bloque
            # si ya había algo acumulado, lo unimos y guardamos
            if buffer:
                result.append(",".join(buffer))
                buffer = []
            buffer.append(p)  # nuevo bloque
        else:
            buffer.append(p)  # continuación del bloque

    # último bloque
    if buffer:
        result.append(",".join(buffer))
        
    result = [f"\"{x}\"" for x in result]

    return result

def compare_history_vs_snapshot(row):
    history = row['A | Top 100 Regions Changes Status - Summary']
    snapshot = row['💯TOP 100 TITLE - LAUNCH']
    
    # Convertir en listas
    hist_entries = [h.strip() for h in history.split(",") if h.strip() != ""]
    snap_entries = [s.strip().replace(" - checked","") for s in snapshot.split(",") if s.strip() != ""]
    
    # Filtrar ruido: descartar los que empiezan con "-" o cuyo código es "-"
    clean_hist_entries = []
    for h in hist_entries:
        if " - " in h:
            code, action = h.split(" - ", 1)
            if code.strip() != "-":
                clean_hist_entries.append(h)
    
    # Extraer códigos del historial ya limpio
    hist_codes = [h.split(" - ")[0] for h in clean_hist_entries if " - " in h]
    
    # Guardamos los cambios en una lista separada
    changes = []
    for code in set(snap_entries) | set(hist_codes):
        if code in snap_entries and code in hist_codes:
            changes.append(f"{code} - Maintained {today}")
        elif code in snap_entries and code not in hist_codes:
            changes.append(f"{code} - Added {today}")
        elif code not in snap_entries and code in hist_codes:
            changes.append(f"{code} - Removed {today}")
    
    # Ahora historial primero, luego cambios
    results = changes + clean_hist_entries
    results = [r for r in (changes + clean_hist_entries) if not r.startswith("- ")]

    
    return ",".join(results)

def compare_history_vs_snapshot_min(row): 
    history = row['A | Season/Feature Validation Comments AV'] 
    snapshot = row['A | Season Active'] 
    # Los convertimos a lista 
    hist_entries = [h.strip() for h in history.split(",") if h.strip() !=""] 
    snap_entries = [s.strip() for s in snapshot.split(",") if s.strip()!=""] 
    changes = [] 
    
    for snap in snap_entries: 
        if snap.__contains__("No Transition"): 
            changes.append(f"{today} - Removed") 
        else: 
            changes.append(F"{today} - {snap}") 
    
    results = changes + hist_entries 
    
    return " | ".join(results)

def expand_end_dates(df, col="Code + 🔒 License end"):
    """
    Expande columna de fechas en columnas separadas conservando la original.
    """
    def extract_dates(text):
        if pd.isna(text) or text.strip() == "-":
            return {}
        codes_dict = {}
        for block in text.split(","):
            block = block.strip()
            if " - " in block:
                code, value = block.split(" - ", 1)
                codes_dict[f"A | P&S {code.strip()} {col}"] = value.strip()
        return codes_dict

    expanded = df[col].apply(extract_dates).apply(pd.Series)
    df = pd.concat([df, expanded], axis=1)
    return df


def expand_secondary_brands(df, col="Code + MAX Secondary Brand/s"):
    """
    Expande columna de códigos de región y marcas en columnas separadas.
    Cambia el nombre de la columna:
        Max BRAND -> Primary Max Brand
        MAX Secondary Brand/s -> Secondary Max Brand
    """
    def extract_codes(text):
        if pd.isna(text) or text.strip() == "-":
            return {}
        text = text.strip().strip('"')
        codes_dict = {}
        for block in text.split(","):
            block = block.strip().strip('"')
            if " - " in block:
                code, value = block.split(" - ", 1)
                code = code.strip()

                # 🔧 Normalizar códigos
                if code == "SWL":
                    code = "SWL"

                # Determinar el nombre final
                base_name = col.replace("Code + ", "")
                if base_name == "Max BRAND":
                    base_name = "Primary Max Brand"
                elif base_name == "MAX Secondary Brand/s":
                    base_name = "Secondary Max Brand"

                codes_dict[f"A | P&S {code} {base_name}"] = value.strip()
        return codes_dict

    expanded = df[col].apply(extract_codes).apply(pd.Series)
    df = pd.concat([df, expanded], axis=1)
    return df



def clean_priority_list(series: pd.Series, col_name: str) -> str:
    """
    Limpia y ordena listas de región-prioridad.
    """
    valid_entries = series.dropna().astype(str).str.strip()
    
    filtered = [entry for entry in valid_entries if not entry.endswith(" - -")]

    return ",".join(sorted(filtered))


def split_region(df: pd.DataFrame, region_name: str, codes: list[str]) -> pd.DataFrame:
    """
    Separa filas con múltiples regiones en varias filas.
    """
    rows = df[df["Region"] == region_name]
    if rows.empty:
        return df
    df = df[df["Region"] != region_name]
    new_rows = [rows.assign(**{"Region Code": code}) for code in codes]
    return pd.concat([df] + new_rows, ignore_index=True)


def update_from_map(df: pd.DataFrame, cols_map: dict) -> pd.DataFrame:
    """
    Actualiza columnas *_x con los valores de *_y según un diccionario de mapeo.
    - Si la columna *_y no existe, conserva *_x sin cambios.
    - Si la columna *_x no existe, se salta.
    """
    for col_x, col_y in cols_map.items():
        if col_x in df.columns:
            if col_y in df.columns:
                print(f"✅ Actualizando: \"{col_x}\" ← \"{col_y}\"")
            
                mask = df[col_y].notnull() & (df[col_x].isnull() | (df[col_x] != df[col_y]))
                df.loc[mask, col_x] = df.loc[mask, col_y]
                df.drop(columns=[col_y], inplace=True)
            else:
                print(f"⚠️  {col_y} no existe → se conserva \"{col_x}\" tal cual")
        else:
            print(f"⏭️  Saltando {col_x} (no existe en df)")
    
    return df


def get_unique_value(pattern: str) -> str:
    # Separar por coma
    parts = [p.strip() for p in pattern.split(",")]
    # Quitar duplicados manteniendo orden
    unique_parts = list(dict.fromkeys(parts))
    # Si hay más de un valor, quedarnos con el último distinto de "-"
    for val in reversed(unique_parts):
        if val != "-":
            return val
    # Si todos son "-", devolver "-"
    return "-"

if __name__=='__main__':
    # ========================
    # Paths y carga de data
    # ========================


    bridge_path = r"C:\Users\azayas\Downloads\Bridge to Update.xlsx"
    main_path = r"C:\Users\azayas\Downloads\P&S Launch List (Pivot)-EMEA W3 Launch List - Ally (44).csv"


    output_path = Path(r"C:\Users\azayas\OneDrive - Warner Bros. Discovery\Documents\Launch List Automation Outputs")

    bridge_df = pd.read_excel(bridge_path, engine="openpyxl")
    main_df = pd.read_csv(main_path)

    # ========================
    # Tratamiento de fechas
    # ========================

    bridge_df['🔒 License end'] = pd.to_datetime(bridge_df['🔒 License end'], format="%d-%b-%y", errors="coerce")


    bridge_df['🔒 License end'] = bridge_df['🔒 License end'].dt.strftime("%d/%m/%Y")

    bridge_df['🔒 License end'] = bridge_df['🔒 License end'].astype(str)

    bridge_df['✏️ Planned Premiere Date'] = pd.to_datetime(bridge_df['✏️ Planned Premiere Date'], format="%d-%b-%y", errors="coerce")


    bridge_df['✏️ Planned Premiere Date'] = bridge_df['✏️ Planned Premiere Date'].dt.strftime("%d/%m/%Y")

    bridge_df['✏️ Planned Premiere Date'] = bridge_df['✏️ Planned Premiere Date'].astype(str)


    bridge_df['👪 Kids Age Demo'] = bridge_df['👪 Kids Age Demo'].str.replace(",", "|", regex=False)
    bridge_df["MAX Secondary Brand/s"] = bridge_df["MAX Secondary Brand/s"].str.replace(",", "|", regex=False)

    bridge_df['🔒 Distributor'] = bridge_df['🔒 Distributor'].str.replace(",", "|", regex=False)

    # ========================
    # Renombrar columnas y mapear regiones
    # ========================
    cols_map = {"🔒 Region": "Region"} 
    # cols_map = {"🔒 Region": "Region", "GR/TR/WAVE 3 Priority": "Priority"}
    bridge_df.rename(columns=cols_map, inplace=True)

    region_code_map = {
        "Adria": "ADR", "Iceland": "ICE", "Baltics": "BAL", "CIS": "CIS",
        "Israel": "ISR", "Malta": "MAL", "Ukraine": "UKR", "Ireland": "IRE",
        "UK": "UK", "Germany": "GAU", "Italy": "ITA", "Luxembourg": "LUX",
        "Switzerland": "SWL",
        "Greece & Cyprus": None, "UK & Ireland": None
    }

    df = bridge_df.copy()
    df["Region"] = df["Region"].str.strip()
    df["GR/TR/WAVE 3 Priority"] = df["GR/TR/WAVE 3 Priority"].fillna("-").astype(str).str.strip()
    df["Region Code"] = df["Region"].map(region_code_map.get)

    df['🔒 Production Year'] = df['🔒 Production Year'].astype("Int64").astype(str)

    #df['🔒 Production Year'] = df['🔒 Production Year'].fillna("-").astype(str).str.strip()

    # ========================
    # Tratamiento para Greece
    # ========================

    ## Llenamos los vacíos con "-"
    df['Greece Converged Scheduling Priority'] = df['Greece Converged Scheduling Priority'].fillna("-")
    df['Greece Converged Scheduling Priority'] = df['Greece Converged Scheduling Priority'].astype(str)
    df['Greece Vodafone Status'] = df['Greece Vodafone Status'].fillna("-")


    # Separar regiones especiales
    region_expansions = {"Greece & Cyprus": ["GRE", "CYP"], "UK & Ireland": ["UK","IRE"]}
    for region, codes in region_expansions.items():
        df = split_region(df, region, codes)   

    # ========================
    # Tratamiento para Transiction to Max
    # ========================
    df['Transition to Max Status'] = df['Transition to Max Status'].fillna("-")

    df['Transition to Max Status'] = df['Transition to Max Status'].replace({"DAY 1 - Launch Catalogue":"For Launch", "-":"No Transition to Max"})

    df['Transition to Max Status'] = np.where(df['Transition to Max Status'].str.contains("Day", na=False), "For Launch", df['Transition to Max Status'])
    #df['Transition to Max Status'] = np.where(df['Transition to Max Status'].str.contains("Remove", na=False), "No Transition to Max", df['Transition to Max Status'])

    # ========================
    # Preparar columnas de código
    # ========================

    same_cols = [
        "Content Pipeline", "GR/TR/WAVE 3 Priority", "Max BRAND", "🏅 EMEA Medal",
        "Primary Genre 1 🎭", "Primary Genre 2 🎭", "Sub-Genre 1 🎭", "Sub-Genre 2 🎭",
        "MAX Secondary Brand/s", "👪 Kids Age Demo", "👩‍👩‍👧‍👧 Kids Content Privacy", "🔒 License end",
        "Region","💯TOP 100 TITLE - LAUNCH", "Transition to Max Status","🔒Country of Origin", "Greece Converged Scheduling Priority",'✏️ Planned Premiere Date','🔒 Production Year',"🔒 Distributor"
        #,"Greece Vodafone Status",
    ]

    cols_main = [
        "A | P&S Launch List Unique ID", "P&S | Title+Season", "P&S | Content Pipeline Summary",
        "P&S | Priority Regions Summary", "A | P&S Primary Max Brand", "A | P&S EMEA Medal",
        "P&S | Primary Genre 1 Summary", "P&S | Primary Genre 2 Summary",
        "P&S | Sub-Genre 1 Summary", "P&S | Sub-Genre 2 Summary", "A | P&S Secondary Max Brand",
        "P&S | Kids Age Demo Summary", "P&S | Kids Content Privacy Summary", 
    ]


    for col in same_cols:
        print(col)
        
        if col == "Region":
            #print(f"Concatenando columna: {col}")
            df[f"Code + {col}"] = df["Region Code"]
        elif col == "Greece Converged Scheduling Priority":
            df[f"Code + {col}"] = df[col]
            
        elif col == "Greece Vodafone Status":
            print(f"Greece: {col}")
            df[f"Code + {col}"] = df[col]
            
        elif col == "🔒 Distributor":
            df[f"Code + {col}"] = df[col]
            
        elif col == '🔒 Production Year':
            df[f"Code + {col}"] = df[col]
        else:
            #print(f"Concatenando columna: {col}")
            df[f"Code + {col}"] = df["Region Code"] + " - " + df[col]

    # ========================
    # Agregar data agrupada
    # ========================

    result = df[["A | P&S Launch List Unique ID", "A | Title + Season #"]].drop_duplicates(
        subset=["A | P&S Launch List Unique ID"]
    )

    for new_col in [f"Code + {c}" for c in same_cols]:
        grouped = (
            df.groupby("A | P&S Launch List Unique ID")[new_col]
            .apply(lambda x: clean_priority_list(x, col_name=new_col))
            .reset_index()
        )
        grouped[new_col] = grouped[new_col].replace("", "-").fillna("-")
        result = result.merge(grouped, on="A | P&S Launch List Unique ID", how="left")

    result = result.loc[:, ~result.columns.duplicated(keep="last")]

    # ========================
    # Expandir Secondary Max Brand y End Dates
    # ========================


    result = expand_secondary_brands(result, col="Code + MAX Secondary Brand/s")

    result = expand_secondary_brands(result, col="Code + Max BRAND")

    result.rename(columns={"Code + 🔒 License end":"End Date"}, inplace=True)

    result = expand_end_dates(result, col="End Date")


    result.rename(columns={"Code + ✏️ Planned Premiere Date":"✏️ Planned Premiere Date"}, inplace=True)

    result = expand_end_dates(result, col="✏️ Planned Premiere Date")

    uk_planned_premiere_name = [c for c in result if c=='A | P&S UK ✏️ Planned Premiere Date']

    columns_planned_premiere_name = [c for c in result if ("✏️ Planned Premiere Date" in c)&(c not in uk_planned_premiere_name)&(c !='✏️ Planned Premiere Date')]

    result.rename(columns=dict(zip(columns_planned_premiere_name,[x[:11] + x[14:] for x in columns_planned_premiere_name])), inplace=True)
    
    ## Cambiamos solo para UK

    result.rename(columns={'A | P&S UK ✏️ Planned Premiere Date':'A | P&S UK Planned Premiere Date'}, inplace=True)

    #result = expand_end_dates(result, col="Code + P&S | Planned Premiere Date")

    ## Regresamos las columnas al nombre original del bridge

    bridge_cols_map = {f"Code + {c}":c for c in same_cols}
    result.rename(mapper=bridge_cols_map, axis=1, inplace=True)

    # ========================
    # Merge con Main y actualizar columnas dinámicamente
    # ========================

    merged = main_df.merge(result, on="A | P&S Launch List Unique ID", how="left")

    merged = update_from_map(merged, cols_map_premiere_date)
    merged = update_from_map(merged, cols_map_secondary)
    merged = update_from_map(merged, cols_map_primary)


    merged = update_from_map(merged, cols_map_end_date)


    ## Tratamos las columnas Greece Converged Scheduling Priority	y Greece Vodafone Status

    merged["Greece Converged Scheduling Priority"] = merged["Greece Converged Scheduling Priority"].fillna("-")
    #merged['Greece Vodafone Status'] = merged['Greece Vodafone Status'].fillna("-")

    merged['Greece Converged Scheduling Priority'] = merged['Greece Converged Scheduling Priority'].apply(get_unique_value)

    merged['🔒 Distributor'] = merged['🔒 Distributor'].fillna("-")

    merged['🔒 Distributor'] = merged['🔒 Distributor'].apply(get_unique_value)
    #merged['Greece Vodafone Status'] = merged['Greece Vodafone Status'].apply(get_unique_value)


    # ========================
    # Actualizamos las demás columnas
    # ========================
    ordered_dict = {
        "P&S | Title+Season": "A | Title + Season #",
        "P&S | Content Pipeline Summary": "Content Pipeline",
        "P&S | Primary Genre 1 Summary": "Primary Genre 1 🎭",
        "P&S | Primary Genre 2 Summary": "Primary Genre 2 🎭",
        "A | P&S Launch List Unique ID": "A | P&S Launch List Unique ID",
        "P&S | Kids Content Privacy Summary": "👩‍👩‍👧‍👧 Kids Content Privacy",
        "P&S | Kids Age Demo Summary": "👪 Kids Age Demo",
        "P&S | Priority Regions Summary": "GR/TR/WAVE 3 Priority",
        "P&S | Sub-Genre 1 Summary": "Sub-Genre 1 🎭",
        "P&S | Sub-Genre 2 Summary": "Sub-Genre 2 🎭",
        "A | P&S Primary Max Brand": "Max BRAND",
        "A | P&S EMEA Medal": "🏅 EMEA Medal",
        "A | Top 100 Regions Changes Status - Summary":"Resultado",
        "P&S | Launch List Validation ALL Regions":"Transition to Max Status",
        "P&S | Top 100 Regions Summary":"💯TOP 100 TITLE - LAUNCH",
        "P&S | Greece Converged Scheduling Priority":"Greece Converged Scheduling Priority",
        "P&S | Distributor":"🔒 Distributor"
        #"P&S | Greece Vodafone Status":"Greece Vodafone Status",
        #"P&S | Content Pipeline": "Content Pipeline",
    }

    ## Regresamos las columnas al nombre original del bridge
    result.rename(mapper=bridge_cols_map, axis=1, inplace=True)

    # ========================
    # Actualizamos las demás columnas
    # ========================


    merged['💯TOP 100 TITLE - LAUNCH'] = merged['💯TOP 100 TITLE - LAUNCH'].fillna("-")
    merged['A | Top 100 Regions Changes Status - Summary'] = merged['A | Top 100 Regions Changes Status - Summary'].fillna("-")

    merged["Resultado"] = merged.apply(compare_history_vs_snapshot, axis=1)

    merged['Resultado'] = merged['Resultado'].map(lambda x:x.replace("- - Addeed", ""))

    merged['Resultado'].replace({f"- - Added {today}":"-"}, inplace=True)


    for main_col, bridge_col in ordered_dict.items():
        print(f"Procesando {main_col} ← {bridge_col}")
        mask_update = merged[bridge_col].notnull() & (
            merged[main_col].isnull() | (merged[main_col] != merged[bridge_col])
        )
        merged.loc[mask_update, main_col] = merged.loc[mask_update, bridge_col]

    new_main_df = merged.copy()


    for col in [c for c in new_main_df.columns if "End Date" in c]:
        new_main_df[col] = pd.to_datetime(new_main_df[col],format="%d/%m/%Y", errors="coerce").dt.date
        
    for col in [c for c in new_main_df.columns if "Premiere" in c]:
        new_main_df[col] = pd.to_datetime(new_main_df[col],format="%d/%m/%Y", errors="coerce").dt.date
        
    #new_main_df.to_excel(output_path / "updated_main_dates_fixed.xlsx", index=False)


    # ========================
    # Guardar archivos
    # ========================



    columnas_deseadas = list(dict.fromkeys(
        list(ordered_dict.keys()) 
        + [c for c in new_main_df.columns if "Secondary" in c] 
        + [c for c in new_main_df.columns if "Primary" in c] 
        + [c for c in new_main_df.columns if "End Date" in c]
        + [c for c in new_main_df.columns if "Premiere" in c]
        +['🔒 Production Year']
        +['A | Season/Feature Validation Comments AV']
    ))


    new_main_df_cols = new_main_df[columnas_deseadas].copy()

    for col in [c for c in new_main_df_cols.columns if "Secondary Max Brand" in c]:
        
        new_main_df_cols.loc[(new_main_df_cols[col] != "-")&(new_main_df_cols[col].notna()),col].map(lambda x: x.replace(" - ",","))
        
    new_main_df_cols.columns = [x.replace("_x","") if x.endswith("_x") else x for x in new_main_df_cols.columns]

    for col in [c for c in new_main_df_cols.columns if "Secondary" in c]:
        mask = new_main_df_cols[col].notna() & (new_main_df_cols[col] != "-")
        #new_main_df_cols.loc[mask, col] = new_main_df_cols.loc[mask, col].map(lambda x: f'\"{x}\"')
        
    ## Cambios repetidas "P&S | Top 100 Regions Summary"

    new_main_df_cols.loc[(new_main_df_cols["P&S | Top 100 Regions Summary"].notna()),"P&S | Top 100 Regions Summary" ] = new_main_df_cols.loc[(new_main_df_cols["P&S | Top 100 Regions Summary"].notna()),"P&S | Top 100 Regions Summary" ].map(lambda x: ",".join(set(x.split(","))))

    # Cambios repetidas "A | P&S EMEA Medal"

    new_main_df_cols.loc[(new_main_df_cols["A | P&S EMEA Medal"].notna()), "A | P&S EMEA Medal"] = new_main_df_cols.loc[(new_main_df_cols["A | P&S EMEA Medal"].notna()), "A | P&S EMEA Medal"].map(lambda x: ",".join(set(x.split(","))))

    aux = bridge_df[["🔒Country of Origin","A | P&S Launch List Unique ID"]].copy()

    aux.drop_duplicates(subset="A | P&S Launch List Unique ID", inplace=True)

    new_main_df_cols = new_main_df_cols.merge(aux, on="A | P&S Launch List Unique ID", how="left")

    new_main_df_cols.rename(columns={'🔒Country of Origin':'A | P&S Country of Origin'}, inplace=True)

    new_main_df_cols.loc[(new_main_df_cols['P&S | Kids Age Demo Summary'].notna())&(new_main_df_cols['P&S | Kids Age Demo Summary']!="-"), 'P&S | Kids Age Demo Summary'] = new_main_df_cols.loc[(new_main_df_cols['P&S | Kids Age Demo Summary'].notna())&(new_main_df_cols['P&S | Kids Age Demo Summary']!="-"), 'P&S | Kids Age Demo Summary'].map(lambda x: ",".join(split_region_string(x)))

    aux_top = bridge_df[['A | P&S Launch List Unique ID','💯TOP 100 TITLE - LAUNCH',"Region"]].copy()

    aux_top = aux_top.loc[(aux_top['💯TOP 100 TITLE - LAUNCH'].notna())&(aux_top['💯TOP 100 TITLE - LAUNCH']=="checked")].copy()
    aux_top['Region Code'] = aux_top['Region'].map(region_code_map.get)

    # Separar regiones especiales
    region_expansions = {"Greece & Cyprus": ["GRE", "CYP"], "UK & Ireland": ["UK","IRE"]}
    for region, codes in region_expansions.items():
        aux_top = split_region(aux_top, region, codes)

    # Separar regiones especiales
    region_expansions = {"Greece & Cyprus": ["GRE", "CYP"], "UK & Ireland": ["UK","IRE"]}
    for region, codes in region_expansions.items():
        aux_top = split_region(aux_top, region, codes)

    aux_top = aux_top.groupby(['A | P&S Launch List Unique ID'], as_index=False).agg({"Region Code":lambda x: ",".join(x.dropna().astype(str))})

    aux_top.drop_duplicates(subset="A | P&S Launch List Unique ID", inplace=True)
    aux_top.rename(columns={'Region Code':"Region_100"}, inplace=True)
    new_main_df_cols = new_main_df_cols.merge(aux_top, on="A | P&S Launch List Unique ID", how="left")
    new_main_df_cols['P&S | Top 100 Regions Summary'] = new_main_df_cols['Region_100']
    new_main_df_cols.drop(columns='Region_100', inplace=True)

    #new_main_df_cols.drop(columns="P&S | Content Pipeline", inplace=True)
    new_main_df_cols.drop(columns=['A | P&S Secondary Max Brand','MAX Secondary Brand/s','A | P&S Primary Max Brand','Primary Genre 1 🎭','Primary Genre 2 🎭',"✏️ Planned Premiere Date","P&S | Planned Premiere Date"], inplace=True)

    new_main_df_cols['P&S | Greece Converged Scheduling Priority'] = new_main_df_cols['P&S | Greece Converged Scheduling Priority'].str.encode("ascii", "ignore").str.decode("ascii")

    greece_aux = bridge_df[['A | P&S Launch List Unique ID','Greece Vodafone Status']].drop_duplicates(subset='A | P&S Launch List Unique ID').copy()
    new_main_df_cols = new_main_df_cols.merge(greece_aux, on='A | P&S Launch List Unique ID', how="left")

    new_main_df_cols.rename(columns={"Greece Vodafone Status":"P&S | Greece Vodafone Status","🔒 Production Year":"P&S | Production Year"}, inplace=True)

    for col in [c for c in new_main_df_cols.columns if "Secondary" in c]:
        new_main_df_cols[col] = new_main_df_cols[col].where(
            new_main_df_cols[col].notna(), 
            other=new_main_df_cols[col]
        ).astype(str).str.replace("|", ",")
        
    new_main_df_cols['P&S | Kids Age Demo Summary'] = new_main_df_cols['P&S | Kids Age Demo Summary'].str.replace("|",",", regex=False)

    #new_main_df_cols["P&S | Production Year"] = (
    #    new_main_df_cols["P&S | Production Year"]
    #    .str.replace(r"[A-Z]{2,3} - <NA>,?", "", regex=True)  # remove patterns
    #    .str.strip(", ")  # clean up leftover commas/spaces
    #)

    new_main_df_cols["P&S | Production Year"].fillna("-", inplace=True)
    new_main_df_cols["P&S | Production Year"] = new_main_df_cols["P&S | Production Year"].str.replace(r"[A-Z]{2,3} - <NA>,?","", regex=True).str.strip(", ").apply(get_unique_value)

    new_main_df_cols['P&S | Production Year'] = new_main_df_cols['P&S | Production Year'].str.replace("<NA>","",regex=False)

    #new_main_df_cols['P&S | Production Year'].replace("<NA>","", inplace=True)
    #new_main_df_cols['P&S | Production Year'].replace("-","",inplace=True)
    #new_main_df_cols['P&S | Production Year'].value_counts()

    cols_premiere = [c for c in new_main_df_cols.columns if "Premiere" in c]

    new_main_df_cols[cols_premiere] = new_main_df_cols[cols_premiere].fillna("-")

    new_main_df_cols[cols_premiere] = new_main_df_cols[cols_premiere].replace("nan", "-")

    col = "P&S | Launch List Validation ALL Regions"



    # Definimos las condiciones
    cond1 = new_main_df_cols[col].fillna("").str.contains("For Launch|Post Launch", na=False)
    cond2 = (
        new_main_df_cols[col].fillna("").str.contains("No Transition to Max", na=False)
        & ~cond1
    )
    cond3 = new_main_df_cols[col].isna() | (new_main_df_cols[col].str.strip() == "")

    # Definimos los valores que asigna cada condición
    choices = [
        "Season Active",        # cond1
        "No Transition to Max", # cond2
        "Not in Bridge"         # cond3
    ]

    # Aplicamos con np.select
    new_main_df_cols["A | Season Active"] = np.select(
        [cond1, cond2, cond3],
        choices,
        default="Season Removed"  # ELSE final
    )

    new_main_df_cols.fillna("-", inplace=True)



    new_main_df_cols['A | Season/Feature Validation Comments AV'] = new_main_df_cols.apply(compare_history_vs_snapshot_min, axis=1)

    #new_main_df_cols['A | Top 100 Regions Changes Status - Summary'].fillna("-", inplace= True)

    new_main_df_cols[
        [c for c in new_main_df_cols.columns if "Date" in c]
    ] = new_main_df_cols[
        [c for c in new_main_df_cols.columns if "Date" in c]
    ].replace("-", pd.NaT)


    new_main_df_cols['P&S | Content Pipeline Summary'] = new_main_df_cols['P&S | Content Pipeline Summary'].replace("-","", regex=False)
    new_main_df_cols['P&S | Launch List Validation ALL Regions'] = new_main_df_cols['P&S | Launch List Validation ALL Regions'].replace("-","",regex=False)


    new_main_df_cols.drop(columns='End Date', inplace=True)

    new_main_df_cols['P&S | Distributor'] = new_main_df_cols['P&S | Distributor'].str.replace("|", ",", regex=False)

    new_main_df_cols.to_csv(output_path / "updated_main.csv", index=False)

    print("Proceso completado ✅")
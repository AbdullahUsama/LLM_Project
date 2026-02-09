"""
Inspect all sheets/tabs in the NUST Bank-Product-Knowledge.xlsx file.
Prints: sheet names, shape, columns, dtypes, nulls, and first few rows per sheet.
"""

import pandas as pd
import os

XLSX_PATH = os.path.join(os.path.dirname(__file__), "..", "NUST Bank-Product-Knowledge.xlsx")

# ── Load every sheet ──────────────────────────────────────────────────────────
xls = pd.ExcelFile(XLSX_PATH, engine="openpyxl")

print("=" * 80)
print(f"FILE : {os.path.basename(XLSX_PATH)}")
print(f"TOTAL SHEETS : {len(xls.sheet_names)}")
print(f"SHEET NAMES  : {xls.sheet_names}")
print("=" * 80)

for sheet in xls.sheet_names:
    df = pd.read_excel(xls, sheet_name=sheet)

    print(f"\n{'─' * 80}")
    print(f"SHEET: {sheet}")
    print(f"{'─' * 80}")
    print(f"  Rows x Cols : {df.shape[0]} rows  x  {df.shape[1]} cols")
    print(f"  Columns     : {list(df.columns)}")
    print()

    # Data types
    print("  DATA TYPES:")
    for col in df.columns:
        print(f"    {col:40s}  ->  {df[col].dtype}")
    print()

    # Null counts
    print("  NULL / MISSING VALUES:")
    for col in df.columns:
        n = df[col].isna().sum()
        pct = n / len(df) * 100 if len(df) > 0 else 0
        print(f"    {col:40s}  ->  {n:>5}  ({pct:.1f}%)")
    print()

    # Unique counts
    print("  UNIQUE VALUES:")
    for col in df.columns:
        print(f"    {col:40s}  ->  {df[col].nunique()}")
    print()

    # Preview first 5 rows
    print("  FIRST 5 ROWS:")
    with pd.option_context("display.max_columns", None, "display.width", 200, "display.max_colwidth", 60):
        print(df.head().to_string(index=False))
    print()

    # Preview last 2 rows
    print("  LAST 2 ROWS:")
    with pd.option_context("display.max_columns", None, "display.width", 200, "display.max_colwidth", 60):
        print(df.tail(2).to_string(index=False))

print("\n" + "=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)

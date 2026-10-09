import os
import pandas as pd

base = r'ThermoDataBase'

print('=== GROUPS AND SUBJECT COUNTS ===')
for group in os.listdir(base):
    group_path = os.path.join(base, group)
    if os.path.isdir(group_path):
        subjects = [s for s in os.listdir(group_path) if os.path.isdir(os.path.join(group_path, s))]
        print(f'{group}: {len(subjects)} subject folders')
        print('  sample names:', subjects[:5])

print('\n=== FULL CONTENTS OF FIRST SUBJECT FOLDER ===')
first_group = [g for g in os.listdir(base) if os.path.isdir(os.path.join(base, g))][0]
first_group_path = os.path.join(base, first_group)
first_subject = [s for s in os.listdir(first_group_path) if os.path.isdir(os.path.join(first_group_path, s))][0]
first_subject_path = os.path.join(first_group_path, first_subject)
for root, dirs, files in os.walk(first_subject_path):
    for f in files:
        print(os.path.join(root, f))

print('\n=== FILE EXTENSION COUNTS (WHOLE DATASET) ===')
ext_counts = {}
for root, dirs, files in os.walk(base):
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
for ext, count in sorted(ext_counts.items(), key=lambda x: -x[1]):
    print(f'{ext}: {count}')

print('\n=== CSV DIMENSIONS CHECK (first 5 samples) ===')
csv_samples = []
for root, dirs, files in os.walk(base):
    for f in files:
        if f.endswith('.csv'):
            csv_samples.append(os.path.join(root, f))
    if len(csv_samples) >= 5:
        break
for path in csv_samples[:5]:
    df = pd.read_csv(path, header=None)
    nonzero = int((df.values != 0).sum())
    print(f'{os.path.basename(path)}: shape={df.shape}, nonzero_pixels={nonzero}')

print('\n=== METADATA EXCEL FILE ===')
xlsx_path = os.path.join(base, 'Plantar Thermogram Database.xlsx')
if os.path.exists(xlsx_path):
    xls = pd.ExcelFile(xlsx_path)
    print('Sheet names:', xls.sheet_names)
    for sheet in xls.sheet_names:
        df = pd.read_excel(xlsx_path, sheet_name=sheet)
        print(f'\n--- Sheet: {sheet} ---')
        print('Columns:', list(df.columns))
        print('Shape:', df.shape)
        print(df.head(3).to_string())
else:
    print('Excel file not found at expected path')

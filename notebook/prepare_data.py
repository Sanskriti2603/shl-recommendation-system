import pandas as pd
import json

# Load JSON
with open('../data/data_shl.json', 'r', encoding='utf-8') as f:
    raw = json.load(f)

# Extract 'assessments' list safely
assessments = raw.get("assessments", [])

# 🛡️ Filter out any bad entries (not dicts)
cleaned = [item for item in assessments if isinstance(item, dict)]

# Log the difference
print(f"Found {len(assessments)} items, using {len(cleaned)} valid dictionaries.")

# Create DataFrame
df = pd.DataFrame.from_records(cleaned)

# Ensure expected columns exist
for col in ['Assessment Name', 'Description', 'Job Title', 'Test Type Categories']:
    if col not in df.columns:
        df[col] = ""

# Combine for embedding
df['combined_text'] = (
    df['Assessment Name'].astype(str) + ' ' +
    df['Description'].fillna('') + ' ' +
    df['Job Title'].fillna('') + ' ' +
    df['Test Type Categories'].astype(str)
)

# Save
df.to_csv('../data/shl_processed.csv', index=False)
print(f"✅ Saved {len(df)} assessments to ../data/shl_processed.csv")

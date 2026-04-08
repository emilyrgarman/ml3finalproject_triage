#%%
# Import Libraries
from pathlib import Path
import pandas as pd
import re

#%%
# Load Data
if (Path('/content').is_dir()) :
    # Running on Google Colab
    datadir = Path('/content/ml3finalproject_triage')
else:
    datadir = Path('ml3finalproject_triage')

train = pd.read_csv(datadir / 'train.csv')
test = pd.read_csv(datadir / 'test.csv')
complaints = pd.read_csv(datadir / 'chief_complaints.csv')
history = pd.read_csv(datadir / 'patient_history.csv')

#%%
SEVERITY_WORDS = ['mild', 'moderate', 'severe',
                  'critical', 'minor', 'major', 'actively',
                  'intermittent', 'in known patient']

def clean_complaint(x):
    x = re.split(r'[,，]', x)[0]
    return re.sub('|'.join(SEVERITY_WORDS), '', x, flags=re.IGNORECASE).strip()

complaints['cc_stripped'] = complaints['chief_complaint_raw'].apply(clean_complaint)

texts = set(complaints['chief_complaint_raw'].tolist())
text_cc = set(complaints['cc_stripped'].tolist())
print(f'Found {len(texts), len(text_cc)} unique complaints in {len(complaints)} rows')
reduced_complaints = complaints.drop_duplicates(
        subset=['chief_complaint_raw', 'chief_complaint_system'], 
        ignore_index=True, 
        keep='first')
print(f'Found {len(reduced_complaints)} unique complaints after deduplication')

# %%
#complaints.sort_values(by='chief_complaint_raw', inplace=True)

joined = pd.merge(train, reduced_complaints, 
                  how='inner', left_on='patient_id', right_on='patient_id'
                  )
print(len(joined))

rejoined = pd.merge(joined, history, 
                  how='inner', left_on='patient_id', right_on='patient_id'
                  )
print(len(rejoined))

# %%
len(set(train['patient_id'].tolist()))

# %%
complaints.sort_values(by=['cc_stripped', 'chief_complaint_raw'], inplace=True)


# %%
s_r_complaints = complaints.drop_duplicates(subset=['cc_stripped','chief_complaint_raw'])
s_r_complaints.drop(columns=['patient_id', 'chief_complaint_system'], inplace=True)
s_r_complaints.to_csv(datadir / 'reduced_complaints.csv', index=False)

# %%

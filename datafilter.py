#%%
# Import Libraries
from pathlib import Path
import numpy as np
import pandas as pd
import re
import seaborn as sns

np.set_printoptions(formatter={'float': lambda x: "{0:0.4f}".format(x)})

#%%
# Load Data
if (Path('/content').is_dir()) :
    # Running on Google Colab
    datadir = Path('/content/ml3finalproject_triage')
else:
    datadir = Path('.')

train = pd.read_csv(datadir / 'train.csv')
test = pd.read_csv(datadir / 'test.csv')
complaints = pd.read_csv(datadir / 'chief_complaints.csv')
history = pd.read_csv(datadir / 'patient_history.csv')

#%%
SEVERITY_WORDS = ['mild', 'moderate', 'severe',
                  'critical', 'minor', 'major', 'actively',
                  'intermittent', 'in known patient']

def clean_complaint(x):
    x = re.split(r'[,\uff0c]', x)[0]
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
# s_r_complaints.to_csv(datadir / 'reduced_complaints.csv', index=False)

# %%
# Checking if reducing complaints impacts ration of acuities
joined1 = pd.merge(train, complaints, 
                  how='left', left_on='patient_id', right_on='patient_id'
                  )
print(len(joined1))

rejoined1 = pd.merge(joined1, history, 
                  how='left', left_on='patient_id', right_on='patient_id'
                  )
print(len(rejoined1))

# %%
complaints_cc = complaints.drop_duplicates(subset=['cc_stripped'])
print(len(complaints_cc))

df_cc = train.merge(complaints_cc[['patient_id', 'cc_stripped']], on='patient_id', how='inner')
df_cc = df_cc.merge(history, on='patient_id', how='inner')
print(len(df_cc))

# %%
triage_acuity_ratios = pd.concat([
    train['triage_acuity'].value_counts(normalize=True).sort_index(),
    df_cc['triage_acuity'].value_counts(normalize=True).sort_index() 
    ],
    keys=['train', 'cc_stripped'],
    axis = 1
    ).reset_index()

triage_acuity_ratios

# %%
# plot bar chart on acuities

ax = sns.barplot(x='triage_acuity', 
                 y='ratio', 
                 hue='dataset',
                 data=pd.melt(triage_acuity_ratios, 
                              id_vars="triage_acuity",
                              var_name="dataset", 
                              value_name="ratio")
                )
ax.set(xlabel='Acuity', ylabel='Ratio')

# %%
df = train.merge(complaints[['patient_id', 'chief_complaint_raw', 'cc_stripped']], on='patient_id', how='inner')
# df = df.merge(history, on='patient_id', how='inner')
print(len(df))

dummies = pd.get_dummies(df, columns=['triage_acuity'], prefix='ta')
acuity_labels = ['ta_1', 'ta_2', 'ta_3', 'ta_4', 'ta_5']
dummies = dummies[['chief_complaint_raw', 'cc_stripped', *acuity_labels]]
dummies.head()

# %%
counts_raw = dummies.groupby(['chief_complaint_raw'])[acuity_labels].sum() 
counts_stripped = dummies.groupby(['cc_stripped'])[acuity_labels].sum() 

counts_raw['sum'] = counts_raw[acuity_labels].sum(axis=1)
counts_raw['max'] = counts_raw[acuity_labels].max(axis=1)
diverged_raw = counts_raw[counts_raw['sum'] != counts_raw['max']]

counts_stripped['sum'] = counts_stripped[acuity_labels].sum(axis=1)
counts_stripped['max'] = counts_stripped[acuity_labels].max(axis=1)
diverged_stripped = counts_stripped[counts_stripped['sum'] != counts_stripped['max']]


# %%


# # Copied from RNNClassifier section
# class_counts = np.bincount(y_tr_aug)
# weights = 1.0 / class_counts
# weights = weights / weights.sum()
# criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(DEVICE))

# %%
import numpy as np
import pandas as pd

# %%
example_definition = [(0, 2), (1, 4), (2, 12), (3, 4)]
example = np.empty(0, dtype=int)
for val, count in example_definition:
    example = np.append(example, np.full(count, val))
print(example)

# %%
class_counts = np.bincount(example)
print(class_counts)
weights = 1.0 / class_counts
weights = weights / weights.sum()
print(weights)
# %%
print(pd.Series(example).value_counts(normalize=False).sort_index())
print(pd.Series(example).value_counts(normalize=True).sort_index())
# %%

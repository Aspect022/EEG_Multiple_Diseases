"""Fix precision/recall key mismatch in _aggregate_results too"""

filepath = r'd:\Projects\AI-Projects\EEG\src\training\research_trainer.py'

with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix the aggregate keys list
old = "metric_keys = ['accuracy', 'f1_macro', 'f1_weighted', 'precision', 'recall',\n                        'specificity', 'sensitivity', 'auc_roc']"
new = "metric_keys = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro',\n                        'specificity', 'sensitivity', 'auc_roc']"

if old in content:
    content = content.replace(old, new)
    print("Fixed aggregate metric_keys")
else:
    # Try with \r\n
    old2 = old.replace('\n', '\r\n')
    if old2 in content:
        content = content.replace(old2, new.replace('\n', '\r\n'))
        print("Fixed aggregate metric_keys (CRLF)")
    else:
        print("WARNING: metric_keys not found!")

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)

print("Done!")

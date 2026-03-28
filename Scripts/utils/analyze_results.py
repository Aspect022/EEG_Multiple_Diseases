import json

data = json.load(open('outputs/results/experiment_summary.json'))
print(f"{'Model':<30} {'Acc':>7} {'F1':>7} {'AUC':>7} {'MCC':>7} {'Kappa':>7} {'Epoch':>5} {'Time':>8}")
print('-' * 100)
for r in data:
    m = r.get('metrics', {})
    name = r.get('name', r.get('experiment','?'))
    acc = m.get('accuracy', 0)
    f1 = m.get('f1_macro', 0)
    auc = m.get('auc_roc', 0)
    mcc = m.get('MCC', 0)
    kappa = m.get('cohens_kappa', 0)
    epoch = m.get('best_epoch', 0)
    dur = r.get('duration_seconds', 0)
    err = r.get('error', None)
    if err:
        print(f"{name:<30} {'ERROR':>7} {'':>7} {'':>7} {'':>7} {'':>7} {'':>5} {dur:>7.0f}s  | {str(err)[:50]}")
    else:
        print(f"{name:<30} {acc:>7.4f} {f1:>7.4f} {auc:>7.4f} {mcc:>7.4f} {kappa:>7.4f} {epoch:>5} {dur:>7.0f}s")

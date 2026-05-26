import os
import csv
from datetime import datetime

def log_experiment(results: dict, filepath: str = "results/canonical_results.csv"):
    """
    Append experiment results to canonical results CSV file.
    
    The schema is:
    run_id,date,task,model,dataset,n_subjects,fold,epoch_best,
    accuracy,balanced_accuracy,macro_f1,weighted_f1,auc_roc,
    alpha_snn_mean,alpha_qnn_mean,
    command,seed,notes
    """
    # Ensure directory exists
    dir_name = os.path.dirname(filepath)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
        
    header = [
        'run_id', 'date', 'task', 'model', 'dataset', 'n_subjects', 'fold', 'epoch_best',
        'accuracy', 'balanced_accuracy', 'macro_f1', 'weighted_f1', 'auc_roc',
        'alpha_snn_mean', 'alpha_qnn_mean',
        'command', 'seed', 'notes'
    ]
    
    file_exists = os.path.exists(filepath)
    
    # Fill in date if missing
    if 'date' not in results or not results['date']:
        results['date'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    # Ensure all header keys exist in results dict (default to empty string or 0.0)
    row_data = {}
    for key in header:
        row_data[key] = results.get(key, '')
        
    with open(filepath, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)

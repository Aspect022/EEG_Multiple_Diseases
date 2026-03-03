import os
import shutil
from pathlib import Path

def create_project_structure():
    """Create the complete project directory structure"""
    
    base_dir = Path(".")
    
    # Define directory structure
    directories = [
        # Data directories
        "data/raw/kaggle_ecg",
        "data/processed/kaggle_ecg/train",
        "data/processed/kaggle_ecg/val",
        "data/processed/kaggle_ecg/test",
        
        # Source code
        "src/data",
        "src/models/baseline",
        "src/models/snn",
        "src/models/transformer",
        "src/models/quantum",
        "src/models/hybrid",
        "src/training",
        "src/evaluation",
        "src/utils",
        
        # Notebooks
        "notebooks",
        
        # Configs
        "configs",
        
        # Experiments tracking
        "experiments/results",
        
        # Outputs
        "outputs/checkpoints",
        "outputs/logs",
        "outputs/figures",
        "outputs/metrics",
        
        # Documentation
        "docs",
        
        # Scripts
        "scripts",
    ]
    
    print("Creating project structure...")
    for directory in directories:
        dir_path = base_dir / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        
        # Create __init__.py for Python packages
        if directory.startswith("src/"):
            init_file = dir_path / "__init__.py"
            init_file.touch()
    
    print("✓ Project structure created successfully!")
    
    # Create .gitignore
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/
env/

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb_checkpoints/

# Data (don't commit large datasets)
data/raw/
data/processed/
*.csv
*.dat
*.hea

# Models and outputs
outputs/checkpoints/*.pth
outputs/checkpoints/*.pt
*.pth
*.pt

# Logs
outputs/logs/
*.log

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Experiment results (too large)
experiments/results/*/*.pth
experiments/results/*/*.pkl
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    
    print("✓ .gitignore created")
    
    return True

if __name__ == "__main__":
    create_project_structure()
"""
Setup script to create the ml_service package structure.
Run this from the ml-service directory: python setup_structure.py

This script will:
1. Remove old directories (app, training, models) that don't match our package structure
2. Create the new ml_service package with proper __init__.py files
3. Create a tests directory
4. Preserve: data/, raw data/, pyproject.toml, requirements.txt
"""

import os
import shutil
from pathlib import Path


def main():
    # Get the directory where this script is located (ml-service/)
    base_dir = Path(__file__).parent.resolve()

    print(f"Working in: {base_dir}\n")

    # Step 1: Remove old directories that don't match our structure
    old_dirs_to_remove = ["app", "training", "models"]

    print("=" * 50)
    print("STEP 1: Cleaning up old structure")
    print("=" * 50)

    for old_dir in old_dirs_to_remove:
        old_path = base_dir / old_dir
        if old_path.exists():
            print(f"  Removing: {old_dir}/")
            shutil.rmtree(old_path)
        else:
            print(f"  Skipping: {old_dir}/ (doesn't exist)")

    # Step 2: Create new package structure
    print("\n" + "=" * 50)
    print("STEP 2: Creating ml_service package structure")
    print("=" * 50)

    # Define the directory structure
    directories = [
        "ml_service",
        "ml_service/api",
        "ml_service/training",
        "ml_service/models",
        "tests",
        "data/jasper",      # Ensure data folders exist
        "data/not_jasper",
    ]

    for dir_path in directories:
        full_path = base_dir / dir_path
        if not full_path.exists():
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"  Created: {dir_path}/")
        else:
            print(f"  Exists:  {dir_path}/")

    # Step 3: Create __init__.py files
    print("\n" + "=" * 50)
    print("STEP 3: Creating __init__.py files")
    print("=" * 50)

    init_locations = [
        "ml_service/__init__.py",
        "ml_service/api/__init__.py",
        "ml_service/training/__init__.py",
        "ml_service/models/__init__.py",
        "tests/__init__.py",
    ]

    for init_file in init_locations:
        full_path = base_dir / init_file
        if not full_path.exists():
            full_path.write_text('"""Package initialization."""\n')
            print(f"  Created: {init_file}")
        else:
            print(f"  Exists:  {init_file}")

    # Step 4: Create placeholder Python files with docstrings
    print("\n" + "=" * 50)
    print("STEP 4: Creating placeholder module files")
    print("=" * 50)

    placeholders = {
        "ml_service/api/main.py": '''"""
FastAPI application for Jasper image classification.

Entry point: jasper-serve
"""

def serve():
    """Start the FastAPI server."""
    # TODO: Implement FastAPI app
    print("Starting Jasper inference server...")


if __name__ == "__main__":
    serve()
''',
        "ml_service/training/train.py": '''"""
Transfer learning training script for Jasper classifier.

Entry point: jasper-train
"""

def main():
    """Run the training pipeline."""
    # TODO: Implement training logic
    print("Starting Jasper model training...")


if __name__ == "__main__":
    main()
''',
        "ml_service/models/classifier.py": '''"""
Model definition and utilities for Jasper classifier.

Uses ResNet50 with transfer learning for binary classification.
"""

# TODO: Implement model loading and inference utilities
''',
    }

    for file_path, content in placeholders.items():
        full_path = base_dir / file_path
        if not full_path.exists():
            full_path.write_text(content)
            print(f"  Created: {file_path}")
        else:
            print(f"  Exists:  {file_path}")

    # Final summary
    print("\n" + "=" * 50)
    print("SETUP COMPLETE!")
    print("=" * 50)
    print("""
New structure:
ml-service/
├── ml_service/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py          ← FastAPI app (jasper-serve)
│   ├── training/
│   │   ├── __init__.py
│   │   └── train.py         ← Training script (jasper-train)
│   └── models/
│       ├── __init__.py
│       └── classifier.py    ← Model utilities
├── tests/
│   └── __init__.py
├── data/
│   ├── jasper/              ← Put Jasper images here
│   └── not_jasper/          ← Put other images here
├── pyproject.toml
├── requirements.txt
└── setup_structure.py       ← This script (you can delete it)

Next steps:
1. Run: pip install -e .
2. Test entry points: jasper-train and jasper-serve
""")


if __name__ == "__main__":
    main()

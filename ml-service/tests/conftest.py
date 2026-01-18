"""
Pytest configuration and shared fixtures.
"""

import sys
from pathlib import Path

# Add ml_service to path for imports
ml_service_path = Path(__file__).parent.parent
sys.path.insert(0, str(ml_service_path))

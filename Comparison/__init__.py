"""
Comparison of models
"""

from .misc.model_enum import Model
from .misc.compute_errors import compute_error_single_shooting, integrate_sol
from .main_scripts.comparison import ComparisonAnalysis, ComparisonParameters
from .prepare_ocp.acrobot import AcrobotOCP

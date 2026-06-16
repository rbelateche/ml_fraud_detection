"""Fraud detection — production-grade real-time scoring service.

Top-level package. Submodules:
- ``config``      : typed settings loaded from env / YAML.
- ``data``        : data acquisition (synthetic + Kaggle), schema, splitting.
- ``eda``         : exploratory data profiling.
- ``models``      : baselines and (Phase 0.5) the model bake-off.
- ``metrics``     : fraud-appropriate evaluation metrics + cost model.
"""

__version__ = "0.1.0"

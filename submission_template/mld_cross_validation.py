"""CLI companion to the historical notebook, using the actual multi-sweep MLD.

Example:
  .venv/Scripts/python.exe submission_template/mld_cross_validation.py --closure

Independent seed families are the folds; this decoder has no fitted weights.
See mld_improvements/EXPERIMENTS.md for the selection/audit protocol.
"""
from mld_improvements.search import main


if __name__ == "__main__":
    main()

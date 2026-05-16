"""Analysis pipeline for Riedl-style multi-agent coordination experiments.

Submodules:
  synthetic   : generate Riedl-format CSVs without any LLM calls
  estimators  : MI / entropy with bias correction
  pid         : partial information decomposition (wraps `dit`)
  tests       : the four tests from Riedl (2026) sec. 2
  nulls       : permutation null distributions
  significance: Wilcoxon + Fisher
  data_io     : load CSVs, apply devs transform, compute macro signal
"""

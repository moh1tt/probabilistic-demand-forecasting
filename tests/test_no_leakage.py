# Cold-start leakage test (spec §4.4).
#
# Asserts, for every cold-start series ID, that no timestep before the
# cold-start cutoff day appears in any training batch. Implemented in
# Phase 4, once the cold-start holdout exists — see PROGRESS.md.

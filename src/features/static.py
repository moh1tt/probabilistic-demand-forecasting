"""Static covariates (spec §5): category, department, store, state.

item_id is deliberately excluded — spec §5 lists only the coarser hierarchy
levels (cat_id/dept_id/store_id/state_id) as static covariates. item_id has
3,049 distinct values relative to 30,490 series, which would push the model
toward memorizing per-item identity rather than sharing statistical
strength across the hierarchy the way a global model is supposed to.
"""

STATIC_CATEGORICALS = ["cat_id", "dept_id", "store_id", "state_id"]

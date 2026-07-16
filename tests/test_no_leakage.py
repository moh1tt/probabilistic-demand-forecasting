"""Cold-start leakage test (spec §4.4).

Two layers, per the spec's own emphasis that this must be "verified
programmatically, not just by design":

1. Unit-level: `truncate_coldstart_history`/`add_coldstart_flag` do what they
   claim on synthetic polars frames.
2. Library-level: build a real `TimeSeriesDataSet` (the same class
   src/models/global_model.py uses) from synthetic data with a designated
   cold-start series, iterate its *entire* training dataloader (every window
   the library is able to construct, not just the final origin), and assert
   that no batch ever exposes an encoder timestep before the cutoff for that
   series. This is the "for every cold-start series ID, no timestep before
   the cutoff appears in any training batch" check spec §4.4 asks for,
   checked against actual library behavior rather than assumed from the
   truncation alone.
"""

import numpy as np
import pandas as pd
import polars as pl

from src.features.coldstart import (
    add_coldstart_flag,
    coldstart_cutoff_day,
    select_coldstart_series,
    truncate_coldstart_history,
)


def test_truncate_coldstart_history_removes_only_designated_series_pre_cutoff_rows():
    df = pl.DataFrame(
        {
            "id": ["COLD", "COLD", "COLD", "WARM", "WARM", "WARM"],
            "d_num": [1, 2, 3, 1, 2, 3],
        }
    )
    out = truncate_coldstart_history(df, coldstart_ids=["COLD"], cutoff_day=2)
    assert out.filter(pl.col("id") == "COLD")["d_num"].to_list() == [2, 3]
    assert out.filter(pl.col("id") == "WARM")["d_num"].to_list() == [1, 2, 3]


def test_add_coldstart_flag_true_only_within_horizon_of_first_visible_day():
    df = pl.DataFrame({"id": ["A"] * 5, "d_num": [10, 11, 12, 13, 14]})
    out = add_coldstart_flag(df, horizon=3).sort("d_num")
    # first visible day is 10: flag true for d_num in {10, 11}, false from 12 on.
    assert out["coldstart_flag"].to_list() == [1, 1, 0, 0, 0]


def test_add_coldstart_flag_uses_each_series_own_first_day_independently():
    df = pl.DataFrame({"id": ["A", "A", "B", "B"], "d_num": [1, 2, 100, 101]})
    out = add_coldstart_flag(df, horizon=2).sort(["id", "d_num"])
    assert out.filter(pl.col("id") == "A")["coldstart_flag"].to_list() == [1, 0]
    assert out.filter(pl.col("id") == "B")["coldstart_flag"].to_list() == [1, 0]


def test_coldstart_cutoff_day_gives_fewer_than_28_visible_days_at_origin():
    origin_day = 1885
    cutoff = coldstart_cutoff_day(origin_day)
    visible_days = origin_day - cutoff + 1
    assert visible_days < 28


def test_select_coldstart_series_is_stratified_and_reproducible():
    ids_cats = pl.DataFrame(
        {
            "id": [f"id_{i}" for i in range(200)],
            "cat_id": (["FOODS"] * 100) + (["HOBBIES"] * 100),
        }
    )
    selected = select_coldstart_series(ids_cats, frac=0.1, seed=44)
    assert 15 <= len(selected) <= 25  # ~10% of 200, both categories represented
    cats = ids_cats.filter(pl.col("id").is_in(selected))["cat_id"].to_list()
    assert "FOODS" in cats and "HOBBIES" in cats
    # reproducible given the same seed
    assert select_coldstart_series(ids_cats, frac=0.1, seed=44) == selected


def _build_synthetic_dataset(cutoff_day: int, min_encoder_length: int, lags: dict | None = None):
    from pytorch_forecasting import TimeSeriesDataSet
    from pytorch_forecasting.data import GroupNormalizer

    rng = np.random.default_rng(0)
    rows = []
    for d in range(1, 401):
        rows.append({"id": "WARM", "cat_id": "A", "d_num": d, "sales": float(rng.poisson(5))})
    for d in range(1, 401):
        rows.append({"id": "COLD", "cat_id": "A", "d_num": d, "sales": float(rng.poisson(5))})
    df = pl.DataFrame(rows)
    df = truncate_coldstart_history(df, coldstart_ids=["COLD"], cutoff_day=cutoff_day)
    df = add_coldstart_flag(df)

    pdf = df.to_pandas()
    pdf["id"] = pdf["id"].astype("category")
    pdf["cat_id"] = pdf["cat_id"].astype("category")

    max_encoder_length = 60
    max_prediction_length = 28
    training_cutoff = pdf["d_num"].max() - max_prediction_length

    training = TimeSeriesDataSet(
        pdf[pdf.d_num <= training_cutoff],
        time_idx="d_num",
        target="sales",
        group_ids=["id"],
        min_encoder_length=min_encoder_length,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=["cat_id"],
        time_varying_known_reals=["coldstart_flag"],
        time_varying_unknown_reals=["sales"],
        lags=lags or {},
        target_normalizer=GroupNormalizer(groups=["id"], center=False),
        add_relative_time_idx=True,
        add_target_scales=True,
        allow_missing_timesteps=True,
    )
    validation = TimeSeriesDataSet.from_dataset(training, pdf, predict=True, stop_randomization=True)
    return training, validation, pdf


def test_no_pre_cutoff_timestep_in_any_training_batch():
    """The library-level check spec §4.4 explicitly asks for: across every
    window the TimeSeriesDataSet's training dataloader can construct (not
    just the final evaluation origin), the cold-start series' encoder never
    reaches back before its cutoff day.
    """
    cutoff_day = 346  # last 55 days visible, same shape as the real Phase 4 config
    training, _, _ = _build_synthetic_dataset(cutoff_day, min_encoder_length=1)

    train_loader = training.to_dataloader(train=True, batch_size=64, num_workers=0)
    min_encoder_start_seen = {}
    for x, _ in train_loader:
        idx_df = training.x_to_index(x)
        decoder_first_day = x["decoder_time_idx"][:, 0]
        encoder_lengths = x["encoder_lengths"]
        encoder_start = decoder_first_day - encoder_lengths
        for i, sid in enumerate(idx_df["id"].tolist()):
            if encoder_lengths[i].item() == 0:
                continue
            start = int(encoder_start[i].item())
            min_encoder_start_seen[sid] = min(min_encoder_start_seen.get(sid, start), start)

    assert min_encoder_start_seen["COLD"] >= cutoff_day
    # sanity: the warm series is untouched and does use pre-cutoff history,
    # so the assertion above is actually exercising the cold-start-specific path.
    assert min_encoder_start_seen["WARM"] < cutoff_day


def test_coldstart_series_survives_with_short_encoder_at_evaluation_origin():
    """Without lowering min_encoder_length, the library's own filter would
    silently drop the cold-start series from the validation set entirely
    (confirmed while designing this — see PROGRESS.md Phase 4 notes). This
    asserts the actual fix (min_encoder_length=1) keeps it present with an
    encoder shorter than the 28-day cold-start horizon, and that no
    pre-cutoff timestep leaks into that evaluation window either.
    """
    cutoff_day = 346
    _, validation, _ = _build_synthetic_dataset(cutoff_day, min_encoder_length=1)

    val_loader = validation.to_dataloader(train=False, batch_size=64, num_workers=0)
    x, _ = next(iter(val_loader))
    idx_df = validation.x_to_index(x)
    cold_pos = idx_df.index[idx_df["id"] == "COLD"].tolist()
    assert len(cold_pos) == 1
    i = cold_pos[0]

    encoder_length = int(x["encoder_lengths"][i].item())
    assert 0 < encoder_length < 28

    decoder_first_day = int(x["decoder_time_idx"][i, 0].item())
    encoder_start_day = decoder_first_day - encoder_length
    assert encoder_start_day >= cutoff_day


def test_target_lag_28_deletes_coldstart_series_entirely_even_with_min_encoder_length_1():
    """Real bug caught while building the Phase 4 pipeline: TimeSeriesDataSet
    unconditionally drops each series' first max(lags) rows before building
    any window (to avoid NaN lag values). With lag-28 (Phase 3's
    TARGET_LAGS), that deletes a cold-start series' entire ~27-day visible
    history — no min_encoder_length fixes this, because there's no encoder
    data left at all. This is why run_deepar_coldstart.py passes lags={}.
    """
    cutoff_day = 346
    _, validation, _ = _build_synthetic_dataset(cutoff_day, min_encoder_length=1, lags={"sales": [7, 28]})
    val_ids = {sid for _, sid in _iter_batch_ids(validation, train=False)}
    assert "COLD" not in val_ids  # confirms the conflict exists...

    # ...and confirms dropping lags (Phase 4's actual fix) resolves it.
    _, validation_no_lags, _ = _build_synthetic_dataset(cutoff_day, min_encoder_length=1, lags={})
    val_ids_no_lags = {sid for _, sid in _iter_batch_ids(validation_no_lags, train=False)}
    assert "COLD" in val_ids_no_lags


def test_coldstart_series_dropped_entirely_without_lowered_min_encoder_length():
    """Documents *why* min_encoder_length is lowered for Phase 4: with the
    Phase 3 default (30), the cold-start series has no valid window at all
    and the library drops it from both training and validation — it would
    silently vanish from the "cold-start" report rather than leak.
    """
    import warnings

    cutoff_day = 346
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        training, validation, _ = _build_synthetic_dataset(cutoff_day, min_encoder_length=30)
    assert any("COLDSTART" not in str(w.message) and "not present in the dataset index" in str(w.message) for w in caught)

    train_ids = {sid for _, sid in _iter_batch_ids(training, train=True)}
    val_ids = {sid for _, sid in _iter_batch_ids(validation, train=False)}
    assert "COLD" not in train_ids
    assert "COLD" not in val_ids


def _iter_batch_ids(dataset, train: bool):
    loader = dataset.to_dataloader(train=train, batch_size=64, num_workers=0)
    for x, _ in loader:
        idx_df = dataset.x_to_index(x)
        for i, sid in enumerate(idx_df["id"].tolist()):
            yield i, sid

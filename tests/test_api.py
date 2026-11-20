"""Tests for the public API -- the one surface outside code is allowed to depend on.

These are contract tests. Everything under `photosvi.models` may be rearranged freely;
what is asserted here is what the command line, the notebook templates and the GUI in
`photosvi-app` are entitled to assume, so a change that breaks one of these breaks
callers we do not control and should be a deliberate release decision.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from photosvi.api import (
    FitSettings,
    Result,
    Session,
    Study,
    fit_session,
    load_session,
)


def _trace(n=800, seed=0):
    rng = np.random.default_rng(seed)
    y = 0.05 * rng.standard_normal(n).astype(np.float32)
    for t in rng.choice(n - 30, 20, replace=False):
        y[t:t + 25] += np.exp(-np.arange(25) / 6).astype(np.float32)
    return y


# ------------------------------------------------------------------- Session
def test_a_session_reports_its_geometry_in_seconds_and_bins():
    s = Session(trace=_trace(1000), fs=25.0, inj_t=500)
    assert s.n_samples == 1000
    assert s.duration_s == 40.0
    assert s.inj_s == 20.0


def test_an_injection_outside_the_recording_is_an_error():
    """Silently clamping would fit a pre/post split that does not exist."""
    with pytest.raises(ValueError, match="outside the recording"):
        Session(trace=_trace(100), fs=25.0, inj_t=500)


def test_a_trace_with_nans_is_refused_at_the_boundary():
    """NaNs reach the loss as NaN and the fit fails thousands of steps later, so the
    complaint belongs here where it still names the cause."""
    bad = _trace(200)
    bad[7] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        Session(trace=bad, fs=25.0, inj_t=100)


def test_an_empty_trace_is_refused():
    with pytest.raises(ValueError, match="empty"):
        Session(trace=np.array([]), fs=25.0, inj_t=0)


def test_a_negative_sampling_rate_is_refused():
    with pytest.raises(ValueError, match="fs must be positive"):
        Session(trace=_trace(100), fs=-25.0, inj_t=10)


def test_metadata_round_trips_through_the_summary():
    s = Session(trace=_trace(100), fs=25.0, inj_t=10, name="m3",
                metadata={"animal": "m3", "drug": "fentanyl"})
    assert s.summary()["metadata"]["drug"] == "fentanyl"


# --------------------------------------------------------------------- Study
def test_one_session_is_a_legitimate_study():
    """Single-recording analysis must not need a different type from a cohort."""
    st = Study([Session(trace=_trace(100), fs=25.0, inj_t=10)])
    assert len(st) == 1
    assert st.summary()["n_sessions"] == 1


def test_a_study_needs_at_least_one_session():
    with pytest.raises(ValueError, match="at least one"):
        Study([])


def test_duplicate_session_names_are_refused():
    """Names index the results, so two sessions called the same thing lose one."""
    s = [Session(trace=_trace(100), fs=25.0, inj_t=10, name="a") for _ in range(2)]
    with pytest.raises(ValueError, match="unique"):
        Study(s)


# --------------------------------------------------------------- FitSettings
def test_settings_round_trip_through_a_dict():
    st = FitSettings(steps=123, tau=0.3)
    assert FitSettings.from_dict(st.to_dict()) == st


def test_every_default_comes_from_the_shipped_config():
    """The regression that matters: this module must not carry its own copy.

    It did, once. `warm.gain` was 1.0 here against 2.0 in the config and `window` was
    8 against 5, and the result was a rate ratio of 1.26 where the truth was 1.78 --
    on data drawn from the model's own generative process, so nothing about the data
    explained it. Values retyped from memory drift silently; this asserts they are
    read.
    """
    import yaml
    cfg = yaml.safe_load(open("configs/final_detection_48.yaml"))
    rd = yaml.safe_load(open("configs/cohort_power_48.yaml"))["fit"]
    r = FitSettings().resolved()
    f, w, d, k = cfg["fit"], cfg["warm"], cfg["detection"], cfg["kernel"]
    assert r["warm"] == w
    assert (r["tau"], r["lr"]) == (f["tau"], f["lr"])
    assert (r["beta_prior"], r["beta_guide"]) == (f["beta_prior"], f["beta_guide"])
    assert r["beta0_prior_scale"] == f["beta0_prior_scale"]
    assert r["beta_inj_prior_scale"] == f["beta_inj_prior_scale"]
    assert (r["mf_thresh"], r["refractory"]) == (d["mf_thresh"], d["refractory"])
    assert r["event_threshold"] == d["thresh"]
    assert (r["kernel"], r["kernel_length"]) == (k["fit"], k["length"])
    assert r["window"] == cfg["guide"]["window"] == f["window"]
    assert r["mark_gate_floor"] == f["mark_gate_floor"]
    # The final model's marks: fixed in noise units, from the config.
    nu = f["mark_noise_units"]
    assert r["mark_noise_units"] == [nu["mu"], nu["sigma"]]
    assert r["architecture"] == "paper_final"
    # The earlier mark prior stays reachable, with its width from the cohort config.
    assert FitSettings(mark_noise_units=[]).resolved()["mark_noise_units"] is None
    assert r["mark_prior_sd"] == rd["mark_prior"]["mu_sd"] == rd["mark_prior"]["log_sigma_sd"]
    # Read off each recording, so nothing is resolved before there is one.
    assert (r["sigma_y"], r["ar_coeffs"], r["mu_a"], r["sigma_a"]) == (None,) * 4
    assert (r["steps"], r["max_steps"]) == (f["stop_min_steps"], f["max_steps"])
    assert r["stop_probe"] == f["stop_probe"]


def test_a_short_run_narrows_the_probe_so_it_can_actually_stop():
    """The plateau is only checked on a probe. Asking for 30 steps and getting 60000
    because the first check lands at 3000 is a surprise; the default is untouched."""
    import yaml
    f = yaml.safe_load(open("configs/final_detection_48.yaml"))["fit"]
    assert FitSettings(steps=30).resolved()["stop_probe"] <= 30
    assert FitSettings().resolved()["stop_probe"] == f["stop_probe"]


def test_the_fit_is_built_the_way_the_benchmark_builds_it():
    """The benchmark script is the validated path; the API must not have its own.

    Compared textually rather than by running both, which would take an hour: what is
    checked is that the same config keys reach the same constructor arguments.
    """
    src = Path("scripts/benchmark_guppy.py").read_text()
    for token in ('thresh=setup.warm["thresh"]', 'gain=setup.warm["gain"]',
                  "window=setup.window", "tau=setup.tau",
                  "**setup.beta_prior_kwargs()"):
        assert token in src, f"benchmark no longer uses {token}; api.py mirrors it"
    api = Path("src/photosvi/api.py").read_text()
    for token in ('thresh=setup.warm["thresh"]', 'gain=setup.warm["gain"]',
                  "window=setup.window", "tau=setup.tau",
                  "**setup.beta_prior_kwargs()"):
        assert token in api, f"api.py no longer mirrors the benchmark's {token}"


def test_an_explicit_setting_overrides_the_shipped_one():
    assert FitSettings(tau=0.42).resolved()["tau"] == 0.42
    assert FitSettings().resolved()["tau"] != 0.42


def test_an_unknown_setting_is_refused_rather_than_ignored():
    """A typo that silently fits with the default is a wrong answer with no warning."""
    with pytest.raises(ValueError, match="unknown settings"):
        FitSettings.from_dict({"stpes": 10})


def test_the_resolved_kernel_names_its_family():
    """build_kernel defaults to `biexp`, a different kernel that accepts a subset of
    these keys, so omitting `family` would silently fit the wrong one."""
    assert FitSettings().resolved()["kernel"]["family"] == "bigamma"


def test_durations_in_bins_are_rescaled_to_the_recordings_rate():
    """The config is at 48 Hz; the kernel support and the refractory window are times."""
    r48, r25 = FitSettings().resolved(fs=48.0), FitSettings().resolved(fs=25.0)
    assert (r48["kernel_length"], r48["refractory"]) == (400, 15)
    assert (r25["kernel_length"], r25["refractory"]) == (208, 8)
    assert FitSettings(kernel_length=50).resolved(fs=25.0)["kernel_length"] == 50


def test_ar2_is_recovered_from_a_simulated_trace():
    """Yule-Walker on the pre-injection window, the paper's procedure."""
    import torch

    from photosvi.models.points import AdditiveEmission, estimate_ar_coeffs
    phi = [1.432, -0.4464]
    torch.manual_seed(0)
    y = AdditiveEmission(sigma_y=1.0, ar_coeffs=phi).sample(torch.zeros(1, 40000))["y"]
    assert estimate_ar_coeffs(y, 30000) == pytest.approx(phi, abs=0.03)
    assert estimate_ar_coeffs(y[:, :20], None) == []


# --------------------------------------------------------------------- loading
def test_a_headed_csv_loads_and_the_injection_may_be_given_in_seconds(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("dff\n" + "\n".join(f"{v:.5f}" for v in _trace(500)))
    s = load_session(p, fs=25.0, inj_s=8.0)
    assert s.n_samples == 500
    assert s.inj_t == 200


def test_a_named_column_can_be_chosen(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("time,dff\n" + "\n".join(f"{i},{v:.5f}"
                                          for i, v in enumerate(_trace(300))))
    assert load_session(p, fs=25.0, inj_s=1.0, column="dff").n_samples == 300


def test_asking_for_a_column_that_is_not_there_says_which_columns_are(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("time,dff\n0,1.0\n1,2.0\n")
    with pytest.raises(ValueError, match="not in"):
        load_session(p, fs=25.0, inj_s=0.0, column="signal")


def test_a_headerless_file_with_several_columns_refuses_to_guess(tmp_path):
    """Picking column 0 would usually fit the timestamps: a ramp, and a wrong answer
    with nothing to signal it."""
    p = tmp_path / "t.csv"
    p.write_text("\n".join(f"{i},{v:.5f}" for i, v in enumerate(_trace(200))))
    with pytest.raises(ValueError, match="no header"):
        load_session(p, fs=25.0, inj_t=10)
    assert load_session(p, fs=25.0, inj_t=10, column=1).n_samples == 200


def test_an_npz_with_several_arrays_refuses_to_guess(tmp_path):
    np.savez(tmp_path / "t.npz", time=np.arange(200.0), dff=_trace(200))
    with pytest.raises(ValueError, match="pass `column`"):
        load_session(tmp_path / "t.npz", fs=25.0, inj_t=10)
    assert load_session(tmp_path / "t.npz", fs=25.0, inj_t=10,
                        column="dff").n_samples == 200


def test_npy_and_npz_load(tmp_path):
    np.save(tmp_path / "t.npy", _trace(200))
    np.savez(tmp_path / "t.npz", trace=_trace(200))
    assert load_session(tmp_path / "t.npy", fs=25.0, inj_t=50).n_samples == 200
    assert load_session(tmp_path / "t.npz", fs=25.0, inj_t=50).n_samples == 200


def test_an_unsupported_file_type_names_what_is_supported(tmp_path):
    p = tmp_path / "t.wav"
    p.write_bytes(b"RIFF")
    with pytest.raises(ValueError, match="unsupported file type"):
        load_session(p, fs=25.0, inj_t=1)


def test_loading_without_an_injection_is_an_error(tmp_path):
    p = tmp_path / "t.npy"
    np.save(p, _trace(100))
    with pytest.raises(ValueError, match="inj_t .*or inj_s|injection"):
        load_session(p, fs=25.0)


def test_a_two_dimensional_file_is_refused_with_advice(tmp_path):
    np.save(tmp_path / "t.npy", np.zeros((3, 100), dtype=np.float32))
    with pytest.raises(ValueError, match="expected one trace"):
        load_session(tmp_path / "t.npy", fs=25.0, inj_t=10)


# ------------------------------------------------------------------- spectrum
def test_the_spectrum_separates_the_two_components():
    """Carrier below 0.01 Hz, transients in a shoulder from 1 Hz up."""
    from photosvi.api import spectrum
    s = Session(trace=_trace(4000), fs=25.0, inj_t=2000)
    sp = spectrum(s)
    assert len(sp["freq"]) == len(sp["psd"])
    assert sp["band"] == [1.0, 10.0]
    assert sp["band_ratio"] > 0


def test_a_low_passed_recording_is_flagged_before_it_is_fitted():
    """The likeliest silent failure on real data: the fit succeeds and finds nothing.

    A trace filtered into the transient band still looks like a good recording in the
    time domain, and the model returns a rate ratio from it without erroring.
    """
    from photosvi.api import spectrum
    from photosvi.data.loader import filter_signal
    raw = _trace(4000)
    assert not spectrum(Session(trace=raw, fs=25.0, inj_t=2000))["suspect_filtered"]
    with pytest.warns(RuntimeWarning):
        killed = filter_signal(raw, cutoff_hz=1.0, fs=25.0)
    sp = spectrum(Session(trace=killed, fs=25.0, inj_t=2000))
    assert sp["suspect_filtered"]
    assert sp["band_ratio"] < 0.02


# --------------------------------------------------------------------- fitting
@pytest.fixture(scope="module")
def fitted() -> Result:
    """One short real fit, shared: the readouts are what is under test, not the fit."""
    s = Session(trace=_trace(1200), fs=25.0, inj_t=600, name="demo",
                metadata={"animal": "m1", "drug": "saline"})
    return fit_session(s, FitSettings(steps=40, max_steps=40, threads=1))


def test_a_fit_returns_a_rate_ratio_with_an_interval(fitted):
    rr = fitted.rate_ratio
    assert rr["lo"] < rr["estimate"] < rr["hi"]
    assert rr["estimate"] == pytest.approx(math.exp(fitted.beta_inj[0]))
    assert rr["level"] == 0.90


def test_the_event_posterior_is_a_probability_per_bin(fitted):
    p = fitted.event_prob
    assert p.shape == (fitted.session.n_samples,)
    assert (p >= 0).all() and (p <= 1).all()


def test_events_are_suppressed_to_one_per_transient(fitted):
    """Adjacent high-probability bins are one transient, not several detections."""
    ev = fitted.events(threshold=0.1)
    assert np.all(np.diff(ev["bins"]) > fitted.settings.resolved(fs=fitted.session.fs)["refractory"])
    np.testing.assert_allclose(ev["times_s"], ev["bins"] / fitted.session.fs)


def test_a_higher_threshold_never_adds_events(fitted):
    assert fitted.events(0.9)["bins"].size <= fitted.events(0.1)["bins"].size


def test_the_rate_readout_is_per_minute_on_each_side_of_the_injection(fitted):
    r = fitted.rate_per_min(threshold=0.1)
    assert set(r) == {"pre", "post"}
    assert all(v >= 0 for v in r.values())


def test_progress_is_reported_while_fitting():
    """A fit is minutes long; anything driving it from a UI needs this channel."""
    seen = []
    s = Session(trace=_trace(600), fs=25.0, inj_t=300)
    fit_session(s, FitSettings(steps=20, max_steps=20, threads=1),
                on_progress=seen.append, probe_every=5)
    assert len(seen) >= 4
    assert {"step", "total", "loss", "rate_ratio", "elapsed"} <= set(seen[-1])
    assert seen[-1]["step"] > seen[0]["step"]


def test_saving_writes_a_run_directory_a_person_can_read(fitted, tmp_path):
    out = fitted.save(tmp_path / "run")
    assert {p.name for p in out.iterdir()} == {"result.json", "events.csv", "curves.npz"}
    d = json.loads((out / "result.json").read_text())
    assert d["session"]["metadata"]["drug"] == "saline"
    assert d["settings"]["steps"] == 40
    header, *rows = (out / "events.csv").read_text().strip().split("\n")
    assert header == "bin,time_s,probability"
    assert len(rows) == d["n_events"]
    z = np.load(out / "curves.npz")
    assert z["trace"].size == z["event_prob"].size == z["tonic"].size


def test_the_result_records_the_device_it_ran_on(fitted):
    """Which device is a fact about the fit: `auto` sends one recording to the CPU
    and a cohort to the GPU, so two runs on one machine legitimately differ."""
    assert fitted.device in ("cpu", "cuda", "cuda:0", "cuda:1")
    assert fitted.to_dict()["device"] == fitted.device


def test_progress_says_where_it_is_running(fitted):
    """A fit already under way must be able to say it, not only when it finishes."""
    seen = []
    s = Session(trace=_trace(400), fs=25.0, inj_t=200)
    fit_session(s, FitSettings(steps=10, max_steps=10, threads=1),
                on_progress=seen.append, probe_every=5)
    assert seen[-1]["device"] == fitted.device


def test_the_result_dict_is_json_serialisable(fitted):
    """It crosses an HTTP boundary, where a stray numpy scalar is a 500."""
    json.dumps(fitted.to_dict())


def test_the_lag_window_covers_the_kernel_peak():
    """A raw-peak detector fires at the kernel's peak; the scorer has to know that.

    GuPPy timestamps the maximum of the trace, so its detections are late by exactly the
    kernel's peak time. `calibrate_lag` subtracts that systematic offset, but only inside
    `max_lag`, which defaulted to 40 bins -- 1.6 s at 25 Hz and 0.83 s at 48. With a
    kernel peaking at 1.23 s the offset fell outside the window, every GuPPy detection
    scored as a miss, and its F1 read 0.13 at a noise level where the matched filter got
    0.86. That is a measurement artefact that would have been published as a claim about
    another group's method, so the window is now derived from the kernel.
    """
    import numpy as np
    import torch

    from photosvi.benchmark import run_benchmark
    from photosvi.models.points import build_kernel

    fs = 48.0
    slow = build_kernel(dict(family="bigamma", shape_fast=20.34, rate_fast=15.65,
                             shape_slow=15.40, rate_slow=3.95), 400, 1 / fs)
    peak = int(slow.argmax())
    assert peak > 40, "this test is pointless unless the peak is past the old window"

    # a clean trace: events at known times, convolved with that kernel, no noise
    T = 30_000
    rng = np.random.default_rng(0)
    onsets = np.sort(rng.choice(np.arange(500, T - 500, 200), size=120, replace=False))
    drive = torch.zeros(1, T)
    drive[0, torch.as_tensor(onsets.copy())] = 5.0
    y = torch.nn.functional.conv1d(
        drive.reshape(1, 1, -1), slow.flip(0).reshape(1, 1, -1), padding=len(slow) - 1
    )[0, :, : T]

    res = run_benchmark(y, onsets, slow, inj_t=T // 2, fs=fs, tol_s=0.35)
    g = res.outcomes["guppy"]
    assert g.lag >= peak - 3, f"lag {g.lag} did not reach the kernel peak {peak}"
    # and with the offset accounted for, a peak detector on a noiseless trace does well
    assert g.metrics["best_f1"] > 0.5, f"F1 {g.metrics['best_f1']} on a noiseless trace"


def test_a_fitted_model_generates_recordings_like_its_input(fitted):
    """The posterior predictive draws have the recording's length and a plausible count."""
    draws = fitted.simulate(n=3, seed=1)
    assert len(draws) == 3
    for d in draws:
        assert d["y"].shape == (fitted.session.n_samples,)
        assert np.isfinite(d["y"]).all() and d["n_events"] >= 0
    assert fitted.to_dict()["fitted"]["sigma_y"] > 0


def test_the_session_mask_reaches_the_likelihood(monkeypatch):
    """Session.mask was stored and shown in the GUI but never passed to the model."""
    import pyro.infer
    seen = []
    orig = pyro.infer.SVI.step

    def spy(self, *args, **kw):
        seen.append(kw.get("obs_mask"))
        return orig(self, *args, **kw)

    monkeypatch.setattr(pyro.infer.SVI, "step", spy)
    keep = np.ones(1200, dtype=bool)
    keep[600:650] = False
    s = Session(trace=_trace(1200), fs=25.0, inj_t=600, mask=keep)
    fit_session(s, FitSettings(steps=3, max_steps=3, threads=1))
    assert seen and seen[0] is not None
    assert int((~seen[0].cpu().numpy()).sum()) == 50


def test_one_architecture_across_the_final_configs():
    """The paper's final model: every fit setting the single-trace model reads is the same
    in the detection and cohort configs, so Table 1 and Table 2 fit one model."""
    import yaml
    a = yaml.safe_load(open("configs/final_detection_48.yaml"))
    b = yaml.safe_load(open("configs/final_cohort_48.yaml"))
    keys = ("architecture", "mark_mode", "mark_noise_units", "mark_gate_floor",
            "beta0_prior_scale", "beta_inj_prior_loc", "beta_inj_prior_scale", "beta_prior",
            "beta_guide", "tau", "window", "lr", "stop_probe", "stop_tol", "stop_window",
            "stop_min_steps", "max_steps")
    assert {k: a["fit"][k] for k in keys} == {k: b["fit"][k] for k in keys}
    assert a["warm"]["thresh"] == b["warm"]["thresh"] and a["warm"]["gain"] == b["warm"]["gain"]
    assert a["detection"]["refractory"] == b["detection"]["refractory"]
    assert a["kernel"]["fit"] == b["cohort"]["kernel"]
    assert a["kernel"]["length"] == b["cohort"]["kernel_length"]
    assert a["emission"]["ar_coeffs"] == b["cohort"]["ar_coeffs"]
    assert a["tonic"]["drift"] == 0.0 and b["cohort"]["tonic_drift"] == 0.0
    # Same SNR definition at the medium level: e^{mu_a} / sigma_y = 1.95 in both.
    import math
    assert abs(math.exp(a["gt"]["mu_a"]) / 3.0 - 1.95) < 0.01
    assert abs(math.exp(b["cohort"]["mu_a"]) / b["cohort"]["sigma_y"] - 1.95) < 0.01
    assert abs(math.exp(a["fit"]["mark_noise_units"]["mu"]) - 1.95) < 0.01

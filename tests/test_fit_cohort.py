"""`fit_cohort`, the table of sessions and the `--kernel` file.

The fit itself is run end to end on a toy cohort by `docs/examples/toy_fit_cohort.py`
(tests/test_docs_examples.py). These check what is built before the first step, where a
mistake would change the model without failing anything: which window each trace is cut
to, which condition is the reference, which levels the design switches on.
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from photosvi.api import CohortSettings, Session, fit_cohort, read_manifest
from photosvi.cli.manifest import kernel_settings
from photosvi.cohort_fit import prepare_cohort

FS = 48.0


def _session(animal, cond, n=48 * 120, inj=48 * 40, seed=0, fs=FS, mask=None):
    rng = np.random.default_rng(seed)
    return Session(trace=rng.normal(0, 0.01, n).astype("float32"), fs=fs, inj_t=inj,
                   name=f"{animal}_{cond}", mask=mask,
                   metadata=dict(animal=animal, condition=cond))


def _crossed():
    return [_session(a, c, seed=i) for i, (a, c) in
            enumerate([("m1", "veh"), ("m1", "drug"), ("m2", "veh"), ("m2", "drug")])]


def test_every_trace_is_cut_to_one_window_around_its_own_injection():
    ss = [_session("m1", "veh", n=6000, inj=2000), _session("m1", "drug", n=7000, inj=2600),
          _session("m2", "veh", n=6500, inj=1900), _session("m2", "drug", n=6000, inj=2100)]
    c = prepare_cohort(ss, CohortSettings(pre_s=None))
    pre = min(s.inj_t for s in ss)
    post = min(s.n_samples - s.inj_t for s in ss)
    assert c["T"] == pre + post and c["coh"]["inj_t"] == [pre] * 4
    for i, s in enumerate(ss):
        np.testing.assert_array_equal(c["coh"]["y"][i].numpy(),
                                      s.trace[s.inj_t - pre:s.inj_t + post])


def test_pre_s_limits_the_window_before_the_injection():
    c = prepare_cohort(_crossed(), CohortSettings(pre_s=10.0))
    assert c["pre"] == int(10 * FS)


def test_the_reference_is_condition_zero():
    c = prepare_cohort(_crossed(), CohortSettings(), reference="veh")
    assert c["conditions"] == ["veh", "drug"]
    assert c["coh"]["drug"] == [0, 1, 0, 1]
    c = prepare_cohort(_crossed(), CohortSettings())          # alphabetical by default
    assert c["conditions"] == ["drug", "veh"]


def test_the_mask_follows_the_cut():
    keep = np.ones(48 * 120, bool)
    keep[48 * 40:48 * 50] = False
    ss = _crossed()
    ss[0].mask = keep
    c = prepare_cohort(ss, CohortSettings(pre_s=20.0))
    m = c["mask"][0].numpy()
    pre = int(20 * FS)
    assert not m[pre:pre + 480].any() and m[:pre].all() and m[pre + 480:].all()
    assert c["mask"][1:].all()


def test_bad_cohorts_are_refused():
    with pytest.raises(ValueError, match="same sampling rate"):
        prepare_cohort([_session("m1", "veh"), _session("m1", "drug", fs=25.0)],
                       CohortSettings())
    with pytest.raises(ValueError, match="two conditions"):
        prepare_cohort([_session("m1", "veh"), _session("m2", "veh")], CohortSettings())
    s = _session("m1", "veh")
    s.metadata.pop("condition")
    with pytest.raises(ValueError, match="no condition"):
        prepare_cohort([s, _session("m2", "drug")], CohortSettings())
    with pytest.raises(ValueError, match="reference"):
        prepare_cohort(_crossed(), CohortSettings(), reference="saline")


def test_drug_is_accepted_for_condition():
    ss = _crossed()
    for s in ss:
        s.metadata["drug"] = s.metadata.pop("condition")
    assert prepare_cohort(ss, CohortSettings())["conditions"] == ["drug", "veh"]


@pytest.mark.parametrize("design, levels", [
    ([("m1", "veh"), ("m1", "drug"), ("m2", "veh"), ("m2", "drug")], ["animal", "cell"]),
    ([("m1", "veh"), ("m2", "drug"), ("m3", "veh"), ("m4", "drug")], ["animal"]),
])
def test_default_levels_are_the_papers_that_the_design_identifies(design, levels):
    ss = [_session(a, c, n=48 * 30, inj=48 * 10, seed=i) for i, (a, c) in enumerate(design)]
    r = fit_cohort(ss, CohortSettings(min_steps=2, max_steps=4, probe_every=2,
                                      device="cpu"), reference="veh")
    assert list(r.design["levels"]) == levels
    out = r.contrast("drug")
    assert out["lo"] < out["estimate"] < out["hi"] and out["level"] == 0.90
    assert set(r.contrasts_) == {("veh", "drug"), ("drug", "veh")}


def test_the_table_reads_relative_paths_and_keeps_extra_columns(tmp_path):
    for name in ("a.csv", "b.csv"):
        np.savetxt(tmp_path / name, np.zeros(48 * 60))
    (tmp_path / "sessions.csv").write_text(
        "path,animal,condition,fs,inj_start,inj_end,dose\n"
        "a.csv,m1,veh,48,20,25,0\n"
        "b.csv,m1,drug,,20,,3\n")
    sessions, infos = read_manifest(tmp_path / "sessions.csv", mask_min_s=10, fs=48)
    assert [s.metadata for s in sessions] == [
        dict(animal="m1", condition="veh", dose="0"),
        dict(animal="m1", condition="drug", dose="3")]
    assert sessions[0].inj_t == 48 * 20
    assert int((~sessions[0].mask).sum()) == 48 * 10          # max(inj_end, 10 s after)
    assert infos[1]["mask_s"] == 10


def test_a_trace_without_a_rate_is_an_error_that_names_the_line(tmp_path):
    np.savetxt(tmp_path / "a.csv", np.zeros(100))
    (tmp_path / "sessions.csv").write_text("path,animal,condition,inj_start\na.csv,m1,veh,1\n")
    with pytest.raises(ValueError, match="line 2"):
        read_manifest(tmp_path / "sessions.csv")


def test_a_kernel_file_is_rescaled_to_the_recording_rate(tmp_path):
    k = tmp_path / "kernel.yaml"
    k.write_text(yaml.safe_dump(dict(kernel=dict(length=400, fit=dict(family="bigamma")),
                                     data=dict(fs=48.0))))
    assert kernel_settings(k, 48.0)["kernel_length"] == 400
    assert kernel_settings(k, 24.0)["kernel_length"] == 200
    assert kernel_settings(k, 24.0)["kernel"] == dict(family="bigamma")

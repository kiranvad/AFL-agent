"""Tests for ElasticKMeansClustering (elastic k-means phase mapping via autophasemap)."""

from collections import namedtuple
from importlib.util import find_spec

import numpy as np
import pytest
import xarray as xr
from scipy.integrate import trapezoid
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from AFL.double_agent.PipelineOp import PipelineOp
from AFL.double_agent.representations import functionspaces as fs
from AFL.double_agent.representations.functionspaces import ElasticKMeansClustering

requires_autophasemap = pytest.mark.skipif(
    find_spec("autophasemap") is None or find_spec("ray") is None,
    reason="autophasemap is an optional dependency",
)

# small settings for the synthetic data set below
FAST = dict(n_init=1, max_iter=5, num_freq=8, n_neighbors=3, num_cpus=2)


def _gauss(t, mu, sigma=0.04):
    return np.exp(-0.5 * ((t - mu) / sigma) ** 2)


def make_two_phase_dataset(n=16, n_domain=50, seed=0):
    """Two phases along a 1D composition line.

    x < 0.5: a single peak whose position shifts with x (differences along the domain);
    x >= 0.5: a fixed double peak. Elastic k-means should group all shifted single
    peaks together, Euclidean k-means does not.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 1, n)
    t = np.linspace(0, 1, n_domain)
    F = np.array(
        [
            _gauss(t, 0.3 + 0.3 * xi) if xi < 0.5 else _gauss(t, 0.35) + 0.8 * _gauss(t, 0.7)
            for xi in x
        ]
    )
    F = F + 0.05 + 1e-3 * rng.standard_normal(F.shape)
    return xr.Dataset(
        {
            "spectra": (("sample", "q"), F),
            "composition": (("sample", "component"), np.stack([x, 1 - x], axis=1)),
            "truth": (("sample",), (x >= 0.5).astype(int)),
        },
        coords={"sample": np.arange(n) * 10, "q": t, "component": ["A", "B"]},
    )


# ----------------------------------------------------------------------------------
# helpers (no optional dependencies needed)
# ----------------------------------------------------------------------------------


@pytest.mark.unit
def test_unit_domain_maps_to_zero_one():
    t = fs._unit_domain(np.linspace(-2.0, 3.0, 11))
    assert t[0] == 0.0 and t[-1] == 1.0
    np.testing.assert_allclose(np.diff(t), 0.1)


@pytest.mark.unit
def test_unit_domain_rejects_non_monotone():
    with pytest.raises(ValueError, match="strictly increasing"):
        fs._unit_domain(np.array([0.0, 2.0, 1.0]))


@pytest.mark.unit
def test_unit_domain_warns_on_non_uniform_grid():
    with pytest.warns(UserWarning, match="not uniformly spaced"):
        fs._unit_domain(np.geomspace(1e-3, 1.0, 30))


@pytest.mark.unit
@pytest.mark.parametrize("method", ["l2", "max", None])
def test_normalize_functions(method):
    t = np.linspace(0, 1, 50)
    F = np.stack([2 * _gauss(t, 0.5), 5 * _gauss(t, 0.3)])
    out = fs._normalize_functions(F, t, method)
    if method == "l2":
        np.testing.assert_allclose(np.sqrt(trapezoid(out**2, t, axis=1)), 1.0)
    elif method == "max":
        np.testing.assert_allclose(out.max(axis=1), 1.0)
    else:
        np.testing.assert_array_equal(out, F)


@pytest.mark.unit
def test_normalize_functions_rejects_bad_input():
    t = np.linspace(0, 1, 5)
    with pytest.raises(ValueError, match="Unknown normalize"):
        fs._normalize_functions(np.ones((2, 5)), t, "area")
    with pytest.raises(ValueError, match="zero everywhere"):
        fs._normalize_functions(np.zeros((2, 5)), t, "l2")


@pytest.mark.unit
def test_minmax_scale_handles_constant_component():
    C = np.array([[0.0, 5.0], [2.0, 5.0], [4.0, 5.0]])
    np.testing.assert_allclose(fs._minmax_scale(C), [[0, 0], [0.5, 0], [1, 0]])


@pytest.mark.unit
@pytest.mark.parametrize(
    "n_samples,n_domain,candidates,match",
    [
        (20, 5, [2], "points along the function domain"),
        (5, 50, [3], "cannot support"),
        (20, 50, [1], "at least 2"),
    ],
)
def test_validate_problem_errors(n_samples, n_domain, candidates, match):
    with pytest.raises(ValueError, match=match):
        fs._validate_problem(n_samples, n_domain, candidates, n_neighbors=3, num_freq=8)


@pytest.mark.unit
def test_validate_problem_neighbors_and_freq():
    with pytest.raises(ValueError, match="n_neighbors"):
        fs._validate_problem(4, 50, [2], n_neighbors=4, num_freq=2)
    with pytest.warns(UserWarning, match="no effect"):
        fs._validate_problem(10, 50, [2], n_neighbors=3, num_freq=30)


_FakeResult = namedtuple("_FakeResult", "templates gam_ik qik_gam fik_gam delta_n dist d_amplitude error")


def _fake_result(labels, n_phases, n_domain=4):
    """autophasemap-like result where every per-phase array encodes its phase index."""
    n = len(labels)
    per_phase = np.tile(np.arange(n_phases, dtype=float), (n, 1))
    per_point = np.repeat(per_phase[:, :, None], n_domain, axis=2)
    return _FakeResult(
        templates=[np.full(n_domain, k, dtype=float) for k in range(n_phases)],
        gam_ik=per_point,
        qik_gam=per_point,
        fik_gam=per_point,
        delta_n=np.asarray(labels),
        dist=per_phase,
        d_amplitude=per_phase,
        error=0.1,
    )


@pytest.mark.unit
def test_phase_reordering_keeps_arrays_consistent():
    res = _fake_result([1, 1, 1, 0, 2, 2], n_phases=3)
    order = fs._canonical_order(res.delta_n, 3)
    out = fs._apply_order(res, order)

    # phase 1 (3 samples) -> 0, phase 2 (2 samples) -> 1, phase 0 (1 sample) -> 2
    np.testing.assert_array_equal(out["labels"], [0, 0, 0, 2, 1, 1])
    for new, old in enumerate([1, 2, 0]):
        np.testing.assert_array_equal(out["templates"][new], old)
        np.testing.assert_array_equal(out["dist"][:, new], old)
        np.testing.assert_array_equal(out["d_amplitude"][:, new], old)
        np.testing.assert_array_equal(out["fik_gam"][:, new], old)
        np.testing.assert_array_equal(out["gam_ik"][:, new], old)


@pytest.mark.unit
def test_distance_to_membership():
    dist = np.array([[0.1, 1.0, 2.0], [3.0, 0.0, 3.0]])
    p = fs._distance_to_membership(dist, temperature=0.1)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)
    np.testing.assert_array_equal(p.argmax(axis=1), dist.argmin(axis=1))
    # higher temperature gives softer memberships
    assert fs._distance_to_membership(dist, 10.0).max() < p.max()


# ----------------------------------------------------------------------------------
# op behavior without running autophasemap
# ----------------------------------------------------------------------------------


@pytest.mark.unit
def test_defaults_and_json_roundtrip():
    op = ElasticKMeansClustering("spectra", "composition", n_phases=[2, 3], lam=0.1)
    assert op.input_variable == ["spectra", "composition"]
    assert op.functions_variable == "spectra"
    assert op.output_prefix == "elastic_kmeans"
    assert op.name == "ElasticKMeansClustering"

    restored = PipelineOp.from_json(op.to_json())
    assert isinstance(restored, ElasticKMeansClustering)
    assert restored.n_phases == [2, 3]
    assert restored.lam == 0.1
    assert restored.components_variable == "composition"


@pytest.mark.unit
def test_missing_dependency_message(monkeypatch):
    monkeypatch.setattr(fs, "_AUTOPHASEMAP_IMPORT_ERROR", ImportError("not installed"))
    op = ElasticKMeansClustering("spectra", "composition", n_phases=2)
    with pytest.raises(RuntimeError, match=r"AFL-agent\[autophasemap\]"):
        op.calculate(make_two_phase_dataset())


@pytest.mark.unit
def test_selects_lowest_bic_and_writes_outputs(monkeypatch):
    """Model selection and output layout, with the clustering stubbed out."""
    bics = {2: 50.0, 3: 10.0, 4: 30.0}
    calls = []

    def fake_fit(data, n_phases, n_init, random_state, fit_kwargs):
        calls.append((n_phases, n_init, fit_kwargs))
        labels = np.arange(data.N) % n_phases
        return _fake_result(labels, n_phases, n_domain=data.n_domain), bics[n_phases]

    monkeypatch.setattr(fs, "_AUTOPHASEMAP_IMPORT_ERROR", None)
    monkeypatch.setattr(fs, "_ray_session", lambda *args: fs.contextlib.nullcontext())
    monkeypatch.setattr(fs, "_fit_n_phases", fake_fit)

    ds = make_two_phase_dataset()
    op = ElasticKMeansClustering(
        "spectra",
        "composition",
        n_phases=[4, 2, 3, 3],
        n_init=2,
        grid_dim=5,
        lam=0.2,
        membership_temperature=1.0,
        store_alignment=True,
    ).calculate(ds)
    out = op.output

    assert [c[0] for c in calls] == [2, 3, 4]  # deduplicated, sorted
    assert calls[0][1] == 2
    assert calls[0][2]["grid_dim"] == 5 and calls[0][2]["lam"] == 0.2

    labels = out["elastic_kmeans_labels"]
    assert labels.attrs["n_phases"] == 3 and labels.attrs["bic"] == 10.0
    np.testing.assert_array_equal(labels["sample"], ds["sample"])
    assert out["elastic_kmeans_templates"].dims == ("elastic_kmeans_phase", "q")
    assert out["elastic_kmeans_distance"].shape == (16, 3)
    assert out["elastic_kmeans_amplitude_distance"].shape == (16, 3)
    assert out["elastic_kmeans_membership"].shape == (16, 3)
    assert out["elastic_kmeans_aligned"].dims == ("sample", "elastic_kmeans_phase", "q")
    assert out["elastic_kmeans_warping"].shape == (16, 3, 50)
    np.testing.assert_array_equal(out["elastic_kmeans_bic"]["elastic_kmeans_n_phases"], [2, 3, 4])
    np.testing.assert_array_equal(out["elastic_kmeans_bic"], [50.0, 10.0, 30.0])

    merged = op.add_to_dataset(ds)
    assert "elastic_kmeans_labels" in merged and "spectra" in merged


@pytest.mark.unit
def test_rejects_nan_and_ambiguous_domain(monkeypatch):
    monkeypatch.setattr(fs, "_AUTOPHASEMAP_IMPORT_ERROR", None)
    ds = make_two_phase_dataset()

    bad = ds.copy(deep=True)
    bad["spectra"][0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        ElasticKMeansClustering("spectra", "composition", n_phases=2).calculate(bad)

    ds3 = ds.assign(cube=(("sample", "q", "component"), np.zeros((16, 50, 2))))
    with pytest.raises(ValueError, match="domain_dim"):
        ElasticKMeansClustering("cube", "composition", n_phases=2).calculate(ds3)


# ----------------------------------------------------------------------------------
# integration with autophasemap
# ----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ray_session():
    import ray

    ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False, ignore_reinit_error=True)
    yield ray
    ray.shutdown()


@requires_autophasemap
@pytest.mark.integration
def test_recovers_two_phases(ray_session):
    ds = make_two_phase_dataset()
    op = ElasticKMeansClustering(
        "spectra",
        "composition",
        n_phases=2,
        membership_temperature=0.1,
        store_alignment=True,
        **FAST,
    ).calculate(ds)
    out = op.output

    labels = out["elastic_kmeans_labels"]
    assert labels.dims == ("sample",)
    assert labels.attrs["n_phases"] == 2
    assert np.isfinite(labels.attrs["bic"])
    assert adjusted_rand_score(ds["truth"], labels) > 0.9  # 1.0 observed
    # phase 0 is the most populated
    counts = np.bincount(labels.values, minlength=2)
    assert counts[0] >= counts[1]

    assert np.all(np.isfinite(out["elastic_kmeans_templates"]))
    assert out["elastic_kmeans_aligned"].shape == (16, 2, 50)
    np.testing.assert_allclose(out["elastic_kmeans_membership"].sum("elastic_kmeans_phase"), 1.0)
    assert "elastic_kmeans_labels" in op.add_to_dataset(ds)


@requires_autophasemap
@pytest.mark.integration
def test_elastic_beats_euclidean_kmeans_on_shifted_peaks(ray_session):
    ds = make_two_phase_dataset()
    op = ElasticKMeansClustering("spectra", "composition", n_phases=2, **FAST).calculate(ds)

    F = ds["spectra"].values
    F = F / np.linalg.norm(F, axis=1, keepdims=True)
    euclidean = KMeans(2, n_init=10, random_state=0).fit_predict(F)

    ari_elastic = adjusted_rand_score(ds["truth"], op.output["elastic_kmeans_labels"])
    ari_euclidean = adjusted_rand_score(ds["truth"], euclidean)
    assert ari_elastic > ari_euclidean + 0.5  # 1.0 vs 0.21 observed


@requires_autophasemap
@pytest.mark.integration
def test_same_seed_gives_same_labels(ray_session):
    ds = make_two_phase_dataset()
    a = ElasticKMeansClustering("spectra", "composition", n_phases=2, seed=1, **FAST).calculate(ds)
    b = ElasticKMeansClustering("spectra", "composition", n_phases=2, seed=1, **FAST).calculate(ds)
    np.testing.assert_array_equal(
        a.output["elastic_kmeans_labels"], b.output["elastic_kmeans_labels"]
    )


@requires_autophasemap
@pytest.mark.integration
def test_does_not_shut_down_external_ray(ray_session):
    op = ElasticKMeansClustering(
        "spectra", "composition", n_phases=2, keep_ray_alive=False, **FAST
    )
    op.calculate(make_two_phase_dataset())
    assert ray_session.is_initialized()


@requires_autophasemap
@pytest.mark.integration
@pytest.mark.slow
def test_pipeline_on_example_dataset(ray_session, example_dataset1):
    from AFL.double_agent import Pipeline
    from AFL.double_agent.Preprocessor import SavgolFilter

    with Pipeline("elastic_kmeans") as pipeline:
        SavgolFilter(
            input_variable="measurement",
            output_variable="log_measurement",
            dim="x",
            npts=100,
            apply_log_scale=True,
        )
        ElasticKMeansClustering(
            functions_variable="log_measurement",
            components_variable="composition",
            domain_dim="log_x",
            n_phases=2,
            n_init=2,
            max_iter=20,
            seed=0,
            num_cpus=2,
        )
    out = pipeline.calculate(example_dataset1)
    # 1.0 observed
    assert adjusted_rand_score(out["ground_truth_labels"], out["elastic_kmeans_labels"]) > 0.9

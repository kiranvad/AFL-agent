"""Peak scoring behavior and integration with serializable pipelines."""

import numpy as np
import pytest
import xarray as xr

from AFL.double_agent import PeakFeatureScore, Pipeline, UnitCubeTransform
from AFL.double_agent.PipelineOp import PipelineOp


def gaussian(x, location, prominence, fwhm):
    return prominence * np.exp(-4.0 * np.log(2.0) * ((x - location) / fwhm) ** 2)


def test_peak_matching_is_independent_of_target_order_and_preserves_samples():
    x = np.linspace(-2.0, 2.0, 1001)
    targets = [
        {"location": 0.7, "prominence": 2.0, "fwhm": 0.25},
        {"location": -0.7, "prominence": 1.0, "fwhm": 0.35},
    ]
    exact = sum(gaussian(x, **peak) for peak in targets)
    shifted = sum(gaussian(x, **{**peak, "location": peak["location"] + 0.15})
                  for peak in targets)
    dataset = xr.Dataset(
        {"current": (("sample", "potential"), np.stack([exact, shifted]))},
        coords={"sample": [4, 7], "potential": x},
    )
    operation = PeakFeatureScore(targets)
    operation.calculate(dataset)
    assert operation.output["score"][0] < 1e-3
    assert operation.output["score"][1] > operation.output["score"][0]
    np.testing.assert_allclose(operation.output["peak_location"][0], [0.7, -0.7], atol=0.01)
    np.testing.assert_allclose(operation.output["peak_prominence"][0], [2.0, 1.0], rtol=0.01)
    np.testing.assert_allclose(operation.output["peak_fwhm"][0], [0.25, 0.35], rtol=0.01)
    np.testing.assert_array_equal(operation.output["score"].sample, [4, 7])


def test_missing_and_extra_peak_penalties_with_physical_axis_variable():
    x = np.linspace(-2.0, 2.0, 501)
    target = {"location": -0.7, "prominence": 1.0, "fwhm": 0.3}
    operation = PeakFeatureScore([target], missing_peak_penalty=4.0, extra_peak_penalty=0.5)
    dataset = xr.Dataset({"current": ("point", np.zeros(x.size)),
                          "potential": ("point", x)})
    operation.calculate(dataset)
    np.testing.assert_allclose(operation.output["score"], [4.0])
    assert np.isnan(operation.output["peak_location"]).all()
    dataset["current"] = ("point", gaussian(x, **target) + gaussian(x, 0.7, 1.0, 0.3))
    operation.calculate(dataset)
    np.testing.assert_allclose(operation.output["score"], [0.5], atol=1e-3)


def test_pipeline_operations_round_trip_and_run_together():
    x = np.linspace(-1.0, 1.0, 201)
    targets = [{"location": 0.0, "prominence": 1.0, "fwhm": 0.3}]
    with Pipeline() as pipeline:
        PeakFeatureScore(targets, output_variable="peak_loss")
        UnitCubeTransform({"A": {"min": 10.0, "max": 20.0}}, "parameters", "unit")
    restored = Pipeline(ops=[PipelineOp.from_json(op.to_json()) for op in pipeline.ops])
    dataset = xr.Dataset(
        {"current": (("sample", "potential"), gaussian(x, **targets[0])[None, :]),
         "parameters": (("sample", "component"), [[15.0]])},
        coords={"sample": [7], "potential": x, "component": ["A"]},
    )
    result = restored.calculate(dataset, disable_progress_bar=True)
    assert restored.ops[0].name == "PeakFeatureScore"
    assert result.peak_loss.item() < 1e-3
    np.testing.assert_allclose(result.unit, [[0.5]])
    assert {"peak_location", "peak_prominence", "peak_fwhm"}.issubset(result.data_vars)


@pytest.mark.parametrize("potential,current,message", [
    (np.linspace(1, 0, 21), np.zeros(21), "strictly increasing"),
    (np.linspace(0, 1, 21), np.full(21, np.nan), "non-finite"),
    (np.linspace(0, 1, 3), np.zeros(3), "not enough samples"),
])
def test_invalid_spectra_raise_clear_errors(potential, current, message):
    operation = PeakFeatureScore([{"location": 0.5, "prominence": 1.0, "fwhm": 0.2}])
    dataset = xr.Dataset({"current": ("potential", current)},
                          coords={"potential": potential})
    with pytest.raises(ValueError, match=message):
        operation.calculate(dataset)


def test_per_sample_domains_and_different_curve_lengths_round_trip():
    target = {"location": -0.5, "prominence": 1e-6, "fwhm": 0.1}
    rows = []
    for sample, size in (("SAM-1", 501), ("SAM-2", 301)):
        x = np.linspace(-1.2, 0.2, size)
        rows.append(xr.Dataset(
            {"voltage": (("sample", "point"), x[None, :]),
             "signal": (("sample", "point"), gaussian(x, **target)[None, :])},
            coords={"sample": [sample], "point": np.arange(size)},
        ))
    dataset = xr.concat(rows, dim="sample", join="outer")
    operation = PeakFeatureScore(
        [target], domain_variable="voltage", codomain_variable="signal",
        peak_distance=10, savgol_window=7,
    )
    restored = PipelineOp.from_json(operation.to_json())
    assert restored.input_variable == ["voltage", "signal"]
    result = Pipeline(ops=[restored]).calculate(dataset, disable_progress_bar=True)
    np.testing.assert_array_equal(result.score.sample, ["SAM-1", "SAM-2"])
    assert (result.score.values < 1e-3).all()
    np.testing.assert_allclose(result.peak_location[:, 0], -0.5, atol=0.005)
    np.testing.assert_allclose(result.peak_fwhm[:, 0], 0.1, rtol=0.02)
    xr.testing.assert_equal(result.smoothened_voltage, dataset.voltage.rename("smoothened_voltage"))
    assert result.smoothened_signal.dims == ("sample", "point")
    assert np.isnan(result.smoothened_signal.sel(sample="SAM-2")[301:]).all()
    for row in rows:
        sample = row.sample.item()
        expected = restored._smooth_curve(row.voltage.values[0], row.signal.values[0])
        np.testing.assert_allclose(
            result.smoothened_signal.sel(sample=sample)[:len(expected)], expected,
        )


def test_smoothed_outputs_preserve_shared_axis_units_and_input():
    x = np.linspace(-1, 1, 201)
    target = {"location": 0.0, "prominence": 1.0, "fwhm": 0.3}
    y = gaussian(x, **target) + 0.02 * np.sin(250 * x)
    dataset = xr.Dataset(
        {"current": ("potential", y, {"units": "A"})},
        coords={"potential": ("potential", x, {"units": "V"})},
    )
    original = dataset.copy(deep=True)
    operation = PeakFeatureScore([target]).calculate(dataset)
    axis = operation.output["smoothened_potential"]
    smoothed = operation.output["smoothened_current"]
    assert axis.dims == smoothed.dims == ("sample", "potential")
    assert axis.attrs["units"] == "V"
    assert smoothed.attrs["units"] == "A"
    np.testing.assert_array_equal(axis.values[0], x)
    np.testing.assert_allclose(smoothed.values[0], operation._smooth_curve(x, y))
    assert np.std(smoothed.values[0] - gaussian(x, **target)) < np.std(y - gaussian(x, **target))
    xr.testing.assert_identical(dataset, original)


@pytest.mark.parametrize("location", [0, 10, -1])
def test_per_sample_domain_rejects_unpaired_missing_values(location):
    x = np.linspace(0, 1, 51)
    y = np.zeros_like(x)
    x[location] = np.nan
    dataset = xr.Dataset({
        "potential": (("sample", "point"), x[None, :]),
        "current": (("sample", "point"), y[None, :]),
    })
    with pytest.raises(ValueError, match="non-finite"):
        PeakFeatureScore(
            [{"location": 0.5, "prominence": 1, "fwhm": 0.1}],
            domain_variable="potential", codomain_variable="current",
        ).calculate(dataset)


def test_per_sample_domain_rejects_interior_paired_missing_values():
    x = np.linspace(0, 1, 51)
    y = np.zeros_like(x)
    x[10] = y[10] = np.nan
    dataset = xr.Dataset({"potential": ("point", x), "current": ("point", y)})
    with pytest.raises(ValueError, match="non-finite"):
        PeakFeatureScore(
            [{"location": 0.5, "prominence": 1, "fwhm": 0.1}],
        ).calculate(dataset)


def test_shared_domain_broadcasts_with_custom_sample_dimension():
    x = np.linspace(-1, 1, 201)
    target = {"location": 0.0, "prominence": 1.0, "fwhm": 0.3}
    dataset = xr.Dataset(
        {"voltage": ("point", x),
         "signal": (("point", "observation"),
                    np.stack([gaussian(x, **target)] * 2, axis=1))},
        coords={"observation": [4, 7]},
    )
    operation = PeakFeatureScore(
        [target], domain_variable="voltage", codomain_variable="signal",
        sample_dim="observation",
    ).calculate(dataset)
    np.testing.assert_array_equal(operation.output["score"].observation, [4, 7])
    assert (operation.output["score"].values < 1e-3).all()

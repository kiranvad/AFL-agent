"""Peak preprocessing and scoring behavior."""

import numpy as np
import pytest
import xarray as xr

from AFL.double_agent import (
    PeakFeatureScore,
    PeakFitPreprocessor,
    Pipeline,
    SavgolFilter,
    SplinePreprocessor,
)
from AFL.double_agent.PipelineOp import PipelineOp


def gaussian(x, location, prominence, fwhm):
    return prominence * np.exp(-4.0 * np.log(2.0) * ((x - location) / fwhm) ** 2)


def feature_dataset(rows):
    values = np.asarray(rows, dtype=float)
    return xr.Dataset(
        {
            "fitted_peak_location": (("sample", "detected_peak"), values[:, :, 0]),
            "fitted_peak_prominence": (("sample", "detected_peak"), values[:, :, 1]),
            "fitted_peak_fwhm": (("sample", "detected_peak"), values[:, :, 2]),
        },
        coords={"sample": np.arange(values.shape[0])},
    )


def test_score_accepts_peak_values_without_spectrum_preprocessing():
    targets = [
        {"location": 0.7, "prominence": 2.0, "fwhm": 0.25},
        {"location": -0.7, "prominence": 1.0, "fwhm": 0.35},
    ]
    dataset = feature_dataset(
        [
            [[-0.7, 1.0, 0.35], [0.7, 2.0, 0.25], [np.nan] * 3],
            [[-0.55, 1.0, 0.35], [np.nan] * 3, [np.nan] * 3],
        ]
    )
    operation = PeakFeatureScore(targets).calculate(dataset)
    assert operation.output["score"][0] == pytest.approx(0)
    assert operation.output["score"][1] > operation.output["score"][0]
    np.testing.assert_allclose(operation.output["peak_location"][0], [0.7, -0.7])


def test_score_round_trips_and_rejects_partial_peak_rows():
    target = [{"location": 0.0, "prominence": 1.0, "fwhm": 0.3}]
    operation = PipelineOp.from_json(PeakFeatureScore(target).to_json())
    assert operation.input_variable == [
        "fitted_peak_location",
        "fitted_peak_prominence",
        "fitted_peak_fwhm",
    ]
    dataset = feature_dataset([[[0.0, 1.0, np.nan]]])
    with pytest.raises(ValueError, match="all three"):
        operation.calculate(dataset)


def test_separate_preprocessors_reproduce_peak_fit_pipeline():
    pytest.importorskip("lmfit")
    x = np.linspace(-1, 1, 401)
    target = {"location": 0.05, "prominence": 1.0, "fwhm": 0.3}
    y = gaussian(x, **target) + 0.01 * np.sin(170 * x)
    dataset = xr.Dataset(
        {"response": ("position", y, {"units": "counts"})},
        coords={"position": ("position", x, {"units": "mm"})},
    )
    pipeline = Pipeline(
        ops=[
            SavgolFilter(
                input_variable="response",
                output_variable="filtered",
                dim="position",
                npts=401,
                window_length=21,
                polyorder=3,
                apply_log_scale=False,
            ),
            SplinePreprocessor(
                input_variable="filtered",
                output_variable="spline",
                dim="sg_position",
                output_dim="spline_position",
                n_points=401,
            ),
            PeakFitPreprocessor(
                input_variable="response",
                domain_variable="position",
                feature_variable="spline",
                feature_dim="spline_position",
                peak_distance=25,
            ),
            PeakFeatureScore([target]),
        ]
    )
    result = pipeline.calculate(dataset, disable_progress_bar=True)
    assert result.fit_status.item() == "success"
    assert result.score.item() < 0.02
    assert abs(result.peak_location.item() - target["location"]) < 0.01
    assert result.fitted_peak_location.attrs["units"] == "mm"
    assert result.fitted_peak_prominence.attrs["units"] == "counts"
    assert np.isfinite(result.fitted_curve).all()


def test_existing_generic_preprocessors_and_peak_fit_support_batches():
    pytest.importorskip("lmfit")
    x = np.linspace(400, 800, 301)
    target = {"location": 600, "prominence": 2, "fwhm": 40}
    dataset = xr.Dataset(
        {
            "intensity": (
                ("sample", "wavelength"),
                np.stack([gaussian(x, **target), gaussian(x, 605, 2, 40)]),
            )
        },
        coords={"sample": ["a", "b"], "wavelength": x},
    )
    pipeline = Pipeline(
        ops=[
            SavgolFilter(
                input_variable="intensity",
                output_variable="filtered",
                dim="wavelength",
                npts=301,
                window_length=21,
                polyorder=3,
                apply_log_scale=False,
            ),
            SplinePreprocessor(
                input_variable="filtered",
                output_variable="spline",
                dim="sg_wavelength",
                output_dim="spline_wavelength",
                sample_dim="sample",
                n_points=301,
            ),
            PeakFitPreprocessor(
                input_variable="intensity",
                domain_variable="wavelength",
                feature_variable="spline",
                feature_dim="spline_wavelength",
            ),
            PeakFeatureScore([target]),
        ]
    )
    result = pipeline.calculate(dataset, disable_progress_bar=True)
    np.testing.assert_allclose(result.peak_location[:, 0], [600, 605], atol=0.2)


@pytest.mark.parametrize("initial_center", [-0.1, 0.02])
def test_fitted_center_stays_near_detected_guess(initial_center, monkeypatch):
    pytest.importorskip("lmfit")
    from lmfit.model import CompositeModel

    x = np.linspace(-0.12, 0.5, 401)
    operation = PeakFitPreprocessor(center_tolerance=0.05)
    detected = np.array([[initial_center, 1.0, 0.1]])

    def check_optimizer_bounds(model, data, params, **kwargs):
        center = params["peak0_center"]
        assert center.min == max(x[0], initial_center - 0.05)
        assert center.max == min(x[-1], initial_center + 0.05)
        raise RuntimeError("bounds verified")

    monkeypatch.setattr(CompositeModel, "fit", check_optimizer_bounds)
    with pytest.raises(RuntimeError, match="bounds verified"):
        operation._fit_curve(
            x,
            gaussian(x, initial_center + 0.06, 1.0, 0.1),
            detected,
            np.array([initial_center - 0.05]),
            np.array([initial_center + 0.05]),
        )

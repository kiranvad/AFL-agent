"""Score precomputed peak features against requested targets."""

from __future__ import annotations

from typing import Self

import numpy as np
import xarray as xr
from scipy.optimize import linear_sum_assignment

from AFL.double_agent.PipelineOp import PipelineOp

__all__ = ["PeakFeatureScore"]


class PeakFeatureScore(PipelineOp):
    """Score precomputed peak locations, prominences, and FWHMs.

    Peak detection, smoothing, and curve fitting live in separate preprocessing
    operations. The three input variables must share ``sample_dim`` and
    ``peak_dim``. A one-dimensional peak array is treated as a single sample.
    Rows containing three NaNs represent padding or missing peaks.
    """

    def __init__(
        self,
        target_peaks: list[dict[str, float]],
        peak_location_variable: str = "fitted_peak_location",
        peak_prominence_variable: str = "fitted_peak_prominence",
        peak_fwhm_variable: str = "fitted_peak_fwhm",
        output_variable: str = "score",
        sample_dim: str = "sample",
        peak_dim: str = "detected_peak",
        weights: dict[str, float] | None = None,
        missing_peak_penalty: float = 3.0,
        extra_peak_penalty: float = 0.25,
        name: str = "PeakFeatureScore",
    ) -> None:
        variables = (
            peak_location_variable,
            peak_prominence_variable,
            peak_fwhm_variable,
        )
        if any(not isinstance(variable, str) or not variable for variable in variables):
            raise ValueError("peak feature variable names must be non-empty strings")
        if not target_peaks:
            raise ValueError("target_peaks must contain at least one peak")

        required = {"location", "prominence", "fwhm"}
        normalized_targets = []
        for index, peak in enumerate(target_peaks):
            missing = required.difference(peak)
            if missing:
                raise ValueError(f"target_peaks[{index}] is missing {sorted(missing)}")
            values = {key: float(peak[key]) for key in required}
            if not np.isfinite(list(values.values())).all():
                raise ValueError(f"target_peaks[{index}] must contain finite values")
            if values["prominence"] <= 0 or values["fwhm"] <= 0:
                raise ValueError("target prominence and fwhm must be positive")
            normalized_targets.append(values)

        normalized_weights = {
            "location": 1.0,
            "prominence": 1.0,
            "fwhm": 1.0,
            **(weights or {}),
        }
        if set(normalized_weights) != required:
            raise ValueError(f"weights may only contain {sorted(required)}")
        if any(
            not np.isfinite(value) or value < 0 for value in normalized_weights.values()
        ):
            raise ValueError("weights must be finite and non-negative")
        penalties = (missing_peak_penalty, extra_peak_penalty)
        if any(not np.isfinite(value) or value < 0 for value in penalties):
            raise ValueError("peak penalties must be finite and non-negative")

        super().__init__(
            name=name,
            input_variable=list(variables),
            output_variable=output_variable,
        )
        self.target_peaks = normalized_targets
        self.peak_location_variable = peak_location_variable
        self.peak_prominence_variable = peak_prominence_variable
        self.peak_fwhm_variable = peak_fwhm_variable
        self.sample_dim = sample_dim
        self.peak_dim = peak_dim
        self.weights = normalized_weights
        self.missing_peak_penalty = float(missing_peak_penalty)
        self.extra_peak_penalty = float(extra_peak_penalty)

    def calculate(self, dataset: xr.Dataset) -> Self:
        variables = (
            self.peak_location_variable,
            self.peak_prominence_variable,
            self.peak_fwhm_variable,
        )
        missing = [variable for variable in variables if variable not in dataset]
        if missing:
            raise ValueError(f"Dataset has no peak feature variables {missing}")

        arrays = []
        for variable in variables:
            values = dataset[variable]
            if self.peak_dim not in values.dims:
                raise ValueError(
                    f"{variable!r} must have peak dimension {self.peak_dim!r}"
                )
            unexpected = set(values.dims).difference({self.sample_dim, self.peak_dim})
            if unexpected:
                raise ValueError(
                    f"{variable!r} has unsupported dimensions {sorted(unexpected)}"
                )
            if self.sample_dim not in values.dims:
                values = values.expand_dims({self.sample_dim: [0]})
            arrays.append(values.transpose(self.sample_dim, self.peak_dim))

        try:
            arrays = list(xr.align(*arrays, join="exact"))
        except ValueError as exc:
            raise ValueError(
                "peak feature variables must have identical coordinates"
            ) from exc

        feature_values = np.stack(
            [np.asarray(values.values, dtype=float) for values in arrays], axis=-1
        )
        partial = np.isnan(feature_values).any(axis=2) & ~np.isnan(feature_values).all(
            axis=2
        )
        if partial.any():
            raise ValueError("each peak must provide all three features or three NaNs")
        invalid = ~np.isfinite(feature_values) & ~np.isnan(feature_values)
        if invalid.any():
            raise ValueError("peak features must be finite or NaN padding")
        finite_rows = np.isfinite(feature_values).all(axis=2)
        invalid_shapes = finite_rows & (
            (feature_values[:, :, 1] <= 0) | (feature_values[:, :, 2] <= 0)
        )
        if invalid_shapes.any():
            raise ValueError("peak prominence and fwhm values must be positive")

        target = np.array(
            [
                [peak["location"], peak["prominence"], peak["fwhm"]]
                for peak in self.target_peaks
            ],
            dtype=float,
        )
        scale = np.column_stack(
            [
                np.maximum(np.abs(target[:, 2]), np.finfo(float).eps),
                np.maximum(np.abs(target[:, 1]), np.finfo(float).eps),
                np.maximum(np.abs(target[:, 2]), np.finfo(float).eps),
            ]
        )
        weight = np.array(
            [self.weights["location"], self.weights["prominence"], self.weights["fwhm"]]
        )
        scores = np.empty(feature_values.shape[0], dtype=float)
        matched = np.full((feature_values.shape[0], len(target), 3), np.nan)

        for sample_index, sample_features in enumerate(feature_values):
            detected = sample_features[finite_rows[sample_index]]
            if len(detected):
                cost = np.sum(
                    weight
                    * ((target[:, None, :] - detected[None, :, :]) / scale[:, None, :])
                    ** 2,
                    axis=2,
                )
                target_indices, detected_indices = linear_sum_assignment(cost)
                matched[sample_index, target_indices] = detected[detected_indices]
                feature_error = float(cost[target_indices, detected_indices].sum())
            else:
                target_indices = np.array([], dtype=int)
                feature_error = 0.0
            scores[sample_index] = (
                feature_error / len(target)
                + self.missing_peak_penalty * (len(target) - len(target_indices))
                + self.extra_peak_penalty * (len(detected) - len(target_indices))
            )

        coords = {
            self.sample_dim: arrays[0][self.sample_dim],
            "target_peak": np.arange(len(target)),
        }
        self.output = {
            self.output_variable: xr.DataArray(
                scores,
                dims=(self.sample_dim,),
                coords={self.sample_dim: arrays[0][self.sample_dim]},
                attrs={"description": "Peak matching loss; lower is better."},
            ),
            "peak_location": xr.DataArray(
                matched[:, :, 0], dims=(self.sample_dim, "target_peak"), coords=coords
            ),
            "peak_prominence": xr.DataArray(
                matched[:, :, 1], dims=(self.sample_dim, "target_peak"), coords=coords
            ),
            "peak_fwhm": xr.DataArray(
                matched[:, :, 2], dims=(self.sample_dim, "target_peak"), coords=coords
            ),
        }
        for output_name, source in (
            ("peak_location", arrays[0]),
            ("peak_prominence", arrays[1]),
            ("peak_fwhm", arrays[2]),
        ):
            if units := source.attrs.get("units"):
                self.output[output_name].attrs["units"] = units
        return self

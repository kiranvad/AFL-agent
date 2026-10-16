"""Smooth spectra and score their peak features against requested targets."""

from __future__ import annotations

from typing import Self

import numpy as np
import xarray as xr
from scipy.interpolate import UnivariateSpline
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks, peak_widths, savgol_filter

from AFL.double_agent.PipelineOp import PipelineOp

__all__ = ["PeakFeatureScore"]


class PeakFeatureScore(PipelineOp):
    """Smooth spectra, extract peak features, and score requested peaks.

    Parameters
    ----------
    target_peaks
        A list of dictionaries containing ``location``, ``prominence``, and
        ``fwhm``.  Detected peaks are assigned one-to-one to requested peaks by
        minimizing their normalized feature error, so target and detected peak
        order do not need to agree.
    output_variable, sample_dim
        Scalar score output and observation dimension. A lower score is better.
    domain_variable, codomain_variable
        Names of the physical axis and spectrum variables. The domain can be
        a shared 1D coordinate or a per-sample array, such as
        ``potential(sample, point)`` paired with ``current(sample, point)``.
        Each sample is scored on its own axis, without resampling. Paired
        trailing NaNs from concatenating shorter curves are trimmed.
    minimum_prominence
        Absolute prominence floor, in the same units as the input current.
        Detection also requires ``prominence_fraction`` times the smoothed
        current range; the larger threshold is used.
    weights
        Optional weights for ``location``, ``prominence``, and ``fwhm``.
    missing_peak_penalty, extra_peak_penalty
        Additive penalties per unmatched requested or detected peak.

    prominence_fraction, peak_distance
        Relative prominence threshold and minimum spacing in samples.
    savgol_window, savgol_order, spline_order, spline_strength, peak_weight
        Settings for the reference Savitzky-Golay and peak-weighted spline
        smoothing procedure, applied independently to each spectrum.

    Notes
    -----
    Target prominence must use the input current units (e.g. amperes for the
    sample datasets); location and FWHM use spectral coordinate units. No unit
    conversion is performed. Smoothing preserves these units.

    FWHM is measured at half prominence and converted from fractional array
    indices to physical coordinates.  In addition to ``score``, the op emits
    matched ``peak_location``, ``peak_prominence``, and ``peak_fwhm`` arrays
    with dimensions ``(sample, target_peak)``.  Missing matches are NaN.
    ``smoothened_<domain_variable>`` contains the physical axes used for
    smoothing, and ``smoothened_<codomain_variable>`` contains the smoothed
    spectra used for peak detection. Both retain the sample and curve
    dimensions, units, and trailing NaN padding. Domain values are unchanged.
    """

    def __init__(
        self,
        target_peaks: list[dict[str, float]],
        domain_variable: str = "potential",
        codomain_variable: str = "current",
        output_variable: str = "score",
        sample_dim: str = "sample",
        minimum_prominence: float = 0.0,
        weights: dict[str, float] | None = None,
        missing_peak_penalty: float = 3.0,
        extra_peak_penalty: float = 0.25,
        prominence_fraction: float = 0.025,
        peak_distance: int = 25,
        savgol_window: int = 21,
        savgol_order: int = 3,
        spline_order: int = 3,
        spline_strength: float = 0.25,
        peak_weight: float = 5.0,
        name: str = "PeakFeatureScore",
    ) -> None:
        for label, variable in (("domain_variable", domain_variable),
                                ("codomain_variable", codomain_variable)):
            if not isinstance(variable, str) or not variable:
                raise ValueError(f"{label} must be a non-empty variable name")
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
        thresholds = (
            minimum_prominence, prominence_fraction,
            missing_peak_penalty, extra_peak_penalty,
        )
        if any(not np.isfinite(value) or value < 0 for value in thresholds):
            raise ValueError("prominence thresholds and penalties must be finite and non-negative")
        if not isinstance(peak_distance, int) or peak_distance < 1:
            raise ValueError("peak_distance must be a positive integer")
        if not isinstance(savgol_order, int) or savgol_order < 0:
            raise ValueError("savgol_order must be a non-negative integer")
        if not isinstance(savgol_window, int) or savgol_window <= savgol_order:
            raise ValueError("savgol_window must be an integer greater than savgol_order")
        if not isinstance(spline_order, int) or not 1 <= spline_order <= 5:
            raise ValueError("spline_order must be an integer between 1 and 5")
        if not np.isfinite(spline_strength) or spline_strength < 0:
            raise ValueError("spline_strength must be finite and non-negative")
        if not np.isfinite(peak_weight) or peak_weight < 1:
            raise ValueError("peak_weight must be finite and at least one")

        super().__init__(
            name=name,
            input_variable=[domain_variable, codomain_variable],
            output_variable=output_variable,
        )
        self.target_peaks = normalized_targets
        self.domain_variable = domain_variable
        self.codomain_variable = codomain_variable
        self.sample_dim = sample_dim
        self.minimum_prominence = float(minimum_prominence)
        self.weights = normalized_weights
        self.missing_peak_penalty = float(missing_peak_penalty)
        self.extra_peak_penalty = float(extra_peak_penalty)
        self.prominence_fraction = float(prominence_fraction)
        self.peak_distance = peak_distance
        self.savgol_window = savgol_window
        self.savgol_order = savgol_order
        self.spline_order = spline_order
        self.spline_strength = float(spline_strength)
        self.peak_weight = float(peak_weight)

    def calculate(self, dataset: xr.Dataset) -> Self:
        for variable in (self.domain_variable, self.codomain_variable):
            if variable not in dataset:
                raise ValueError(f"Dataset has no variable {variable!r}")
        domain = dataset[self.domain_variable]
        spectra = dataset[self.codomain_variable]
        curve_dims = [dim for dim in spectra.dims if dim != self.sample_dim]
        if len(curve_dims) != 1:
            raise ValueError(f"{self.codomain_variable!r} must have exactly one curve dimension")
        curve_dim = curve_dims[0]
        if set(domain.dims) not in ({curve_dim}, {self.sample_dim, curve_dim}):
            raise ValueError(
                f"{self.domain_variable!r} must use dimension {curve_dim!r} "
                f"with an optional {self.sample_dim!r} dimension"
            )
        domain, spectra = xr.broadcast(domain, spectra)
        if self.sample_dim not in spectra.dims:
            domain = domain.expand_dims({self.sample_dim: [0]})
            spectra = spectra.expand_dims({self.sample_dim: [0]})
        domain = domain.transpose(self.sample_dim, curve_dim)
        spectra = spectra.transpose(self.sample_dim, curve_dim)

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
        scores = np.empty(spectra.sizes[self.sample_dim], dtype=float)
        matched = np.full((len(scores), len(target), 3), np.nan, dtype=float)
        smoothed_values = np.full(spectra.shape, np.nan, dtype=float)

        for sample_index, (x, y) in enumerate(zip(
            np.asarray(domain.values, dtype=float),
            np.asarray(spectra.values, dtype=float),
        )):
            # xr.concat pads shorter curves with paired NaNs on the point axis.
            # Trim only trailing padding; invalid measured points still fail.
            paired_nan = np.isnan(x) & np.isnan(y)
            end = len(x)
            while end and paired_nan[end - 1]:
                end -= 1
            x, y = x[:end], y[:end]
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError(
                    f"spectrum at sample {sample_index} contains non-finite values"
                )
            if x.size < 3:
                raise ValueError("spectral coordinates must be a finite 1D array of length >= 3")
            if not np.all(np.diff(x) > 0):
                raise ValueError("spectral coordinates must be strictly increasing")
            y = self._smooth_curve(x, y)
            smoothed_values[sample_index, :end] = y
            indices, properties = find_peaks(
                y,
                prominence=max(self.minimum_prominence, self.prominence_fraction * np.ptp(y)),
                distance=self.peak_distance,
            )
            if len(indices):
                widths = peak_widths(
                    y,
                    indices,
                    rel_height=0.5,
                    prominence_data=(
                        properties["prominences"],
                        properties["left_bases"],
                        properties["right_bases"],
                    ),
                )
                left = np.interp(widths[2], np.arange(x.size), x)
                right = np.interp(widths[3], np.arange(x.size), x)
                detected = np.column_stack(
                    [x[indices], properties["prominences"], right - left]
                )
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
                detected = np.empty((0, 3))
                feature_error = 0.0

            scores[sample_index] = (
                feature_error / len(target)
                + self.missing_peak_penalty * (len(target) - len(target_indices))
                + self.extra_peak_penalty * (len(detected) - len(target_indices))
            )

        coords = {
            self.sample_dim: spectra[self.sample_dim],
            "target_peak": np.arange(len(target)),
        }
        self.output = {
            f"smoothened_{self.domain_variable}": domain.copy(deep=True),
            f"smoothened_{self.codomain_variable}": spectra.copy(data=smoothed_values),
            self.output_variable: xr.DataArray(
                scores,
                dims=(self.sample_dim,),
                coords={self.sample_dim: spectra[self.sample_dim]},
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
        return self

    def _smooth_curve(
        self, potential: np.ndarray, current: np.ndarray
    ) -> np.ndarray:
        """Smooth current data with Savitzky-Golay filtering, then a spline.

        ``potential`` and ``current`` must be finite one-dimensional arrays of the
        same length, and potential must be strictly increasing. Current units are
        preserved in the returned array.
        """
        savgol_window = self.savgol_window
        savgol_order = self.savgol_order
        spline_order = self.spline_order
        spline_strength = self.spline_strength
        peak_weight = self.peak_weight

        potential = np.asarray(potential, dtype=float)
        current = np.asarray(current, dtype=float)

        if potential.ndim != 1 or current.ndim != 1:
            raise ValueError("potential and current must be one-dimensional")
        if potential.size != current.size:
            raise ValueError("potential and current must have the same length")
        if potential.size <= max(savgol_order, spline_order):
            raise ValueError("not enough samples for the requested filter orders")
        if not np.all(np.isfinite(potential)) or not np.all(np.isfinite(current)):
            raise ValueError("potential and current must contain only finite values")
        if not np.all(np.diff(potential) > 0):
            raise ValueError("potential must be strictly increasing")
        if savgol_window <= savgol_order:
            raise ValueError("savgol_window must be greater than savgol_order")
        if spline_strength < 0:
            raise ValueError("spline_strength must be non-negative")
        if peak_weight < 1:
            raise ValueError("peak_weight must be at least one")

        # Savitzky-Golay requires an odd window no longer than the input. Reduce
        # the requested window for short curves while retaining an odd length.
        maximum_window = potential.size if potential.size % 2 else potential.size - 1
        window = min(savgol_window, maximum_window)
        if window % 2 == 0:
            window -= 1
        if window <= savgol_order:
            raise ValueError("not enough samples for the Savitzky-Golay window")

        savgol_current = savgol_filter(
            current, window_length=window, polyorder=savgol_order
        )

        # The raw-minus-filtered residual isolates high-frequency measurement
        # noise more reliably than differences in the filtered signal. Its robust
        # standard deviation sets a unit-independent spline penalty. Increase
        # spline_strength for an even smoother result or use zero to interpolate
        # the Savitzky-Golay output exactly.
        residual = current - savgol_current
        centered_residual = residual - np.median(residual)
        noise_sigma = np.median(np.abs(centered_residual)) / 0.6745
        smoothing_factor = current.size * noise_sigma**2 * spline_strength

        # Give broad, significant Savitzky-Golay peaks more influence on the
        # spline. This permits strong baseline smoothing without flattening a
        # narrow analytical peak or chasing its single highest noisy sample.
        weights = np.ones(current.size)
        candidate_peaks, _ = find_peaks(
            savgol_current,
            prominence=0.05 * np.ptp(savgol_current),
            distance=window,
        )
        peak_radius = max(2, window // 3)
        for peak in candidate_peaks:
            left = max(0, peak - peak_radius)
            right = min(current.size, peak + peak_radius + 1)
            weights[left:right] = peak_weight

        spline = UnivariateSpline(
            potential,
            savgol_current,
            k=spline_order,
            s=smoothing_factor,
            w=weights,
        )
        return spline(potential)

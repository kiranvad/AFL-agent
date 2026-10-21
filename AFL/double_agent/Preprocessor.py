"""PipelineOps for Data Preprocessing

This module contains preprocessing operations that transform, normalize, and prepare data for analysis.
Preprocessors handle tasks such as:
- Scaling and normalizing data
- Transforming between coordinate systems
- Filtering and smoothing signals
- Extracting features from raw measurements
- Converting between different data representations

Each preprocessor is implemented as a PipelineOp that can be composed with others in a processing pipeline.
"""

import warnings
from numbers import Number
from typing import Union, Optional, List

import numpy as np
import sympy
import xarray as xr
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks, peak_widths, savgol_filter  # type: ignore
from typing_extensions import Self

from AFL.double_agent.PipelineOp import PipelineOp
from AFL.double_agent.util import listify


def _curve_data(
    dataset: xr.Dataset,
    domain_variable: str,
    signal_variable: str,
    sample_dim: str,
) -> tuple[xr.DataArray, xr.DataArray, str]:
    """Broadcast a curve axis and signal to ``(sample, point)`` form."""
    for variable in (domain_variable, signal_variable):
        if variable not in dataset:
            raise ValueError(f"Dataset has no variable {variable!r}")
    domain = dataset[domain_variable]
    signal = dataset[signal_variable]
    curve_dims = [dim for dim in signal.dims if dim != sample_dim]
    if len(curve_dims) != 1:
        raise ValueError(f"{signal_variable!r} must have exactly one curve dimension")
    curve_dim = curve_dims[0]
    if set(domain.dims) not in ({curve_dim}, {sample_dim, curve_dim}):
        raise ValueError(
            f"{domain_variable!r} must use dimension {curve_dim!r} "
            f"with an optional {sample_dim!r} dimension"
        )
    domain, signal = xr.broadcast(domain, signal)
    if sample_dim not in signal.dims:
        domain = domain.expand_dims({sample_dim: [0]})
        signal = signal.expand_dims({sample_dim: [0]})
    return (
        domain.transpose(sample_dim, curve_dim),
        signal.transpose(sample_dim, curve_dim),
        curve_dim,
    )


def _trim_curve(x: np.ndarray, *signals: np.ndarray) -> tuple[int, list[np.ndarray]]:
    """Trim trailing concat padding and validate one sample's curve data."""
    paired_nan = np.isnan(x)
    for signal in signals:
        paired_nan &= np.isnan(signal)
    end = len(x)
    while end and paired_nan[end - 1]:
        end -= 1
    values = [np.asarray(value[:end], dtype=float) for value in (x, *signals)]
    if not all(np.isfinite(value).all() for value in values):
        raise ValueError("curve contains non-finite values")
    if end < 3:
        raise ValueError("curve coordinates must have at least three values")
    if not np.all(np.diff(values[0]) > 0):
        raise ValueError("curve coordinates must be strictly increasing")
    return end, values

class Preprocessor(PipelineOp):
    """Base class stub for all preprocessors

    Parameters
    ----------
    input_variable : str
        The name of the `xarray.Dataset` data variable to extract from the input dataset
    output_variable : str
        The name of the variable to be inserted into the `xarray.Dataset` by this `PipelineOp`
    name : str
        The name to use when added to a Pipeline. This name is used when calling Pipeline.search()
    """

    def __init__(
        self,
        input_variable: str = None,
        output_variable: str = None,
        name: str = "PreprocessorBase",
    ) -> None:
        super().__init__(name=name, input_variable=input_variable, output_variable=output_variable)

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this `PipelineOp` to the supplied `xarray.dataset`"""
        return NotImplementedError(".calculate must be implemented in subclasses")  # type: ignore


class PeakSmoothingPreprocessor(Preprocessor):
    """Apply the peak-preserving smoothing used by the original score op.

    Each sample is processed on its native domain. A Savitzky-Golay pass first
    estimates the high-frequency residual. Its robust standard deviation sets
    a sample-specific spline penalty, while neighborhoods around significant
    peaks receive extra weight so the spline does not flatten them.
    """

    def __init__(
        self,
        input_variable: str = "signal",
        domain_variable: str = "x",
        output_variable: str = "smoothened_signal",
        sample_dim: str = "sample",
        savgol_window: int = 21,
        savgol_order: int = 3,
        spline_order: int = 3,
        spline_strength: float = 0.25,
        peak_weight: float = 5.0,
        candidate_prominence_fraction: float = 0.05,
        name: str = "PeakSmoothingPreprocessor",
    ) -> None:
        if not isinstance(savgol_order, int) or savgol_order < 0:
            raise ValueError("savgol_order must be a non-negative integer")
        if not isinstance(savgol_window, int) or savgol_window <= savgol_order:
            raise ValueError("savgol_window must be greater than savgol_order")
        if not isinstance(spline_order, int) or not 1 <= spline_order <= 5:
            raise ValueError("spline_order must be between one and five")
        if not np.isfinite(spline_strength) or spline_strength < 0:
            raise ValueError("spline_strength must be finite and non-negative")
        if not np.isfinite(peak_weight) or peak_weight < 1:
            raise ValueError("peak_weight must be finite and at least one")
        if (
            not np.isfinite(candidate_prominence_fraction)
            or candidate_prominence_fraction < 0
        ):
            raise ValueError(
                "candidate_prominence_fraction must be finite and non-negative"
            )
        super().__init__(
            name=name,
            input_variable=[domain_variable, input_variable],
            output_variable=output_variable,
        )
        self.domain_variable = domain_variable
        self.signal_variable = input_variable
        self.sample_dim = sample_dim
        self.savgol_window = savgol_window
        self.savgol_order = savgol_order
        self.spline_order = spline_order
        self.spline_strength = float(spline_strength)
        self.peak_weight = float(peak_weight)
        self.candidate_prominence_fraction = float(candidate_prominence_fraction)

    def calculate(self, dataset: xr.Dataset) -> Self:
        domain, signal, _ = _curve_data(
            dataset, self.domain_variable, self.signal_variable, self.sample_dim
        )
        smoothed_values = np.full(signal.shape, np.nan, dtype=float)
        for sample_index, (x, y) in enumerate(zip(domain.values, signal.values)):
            end, (x, y) = _trim_curve(x, y)
            if end <= max(self.savgol_order, self.spline_order):
                raise ValueError("not enough samples for the requested filter orders")
            maximum_window = end if end % 2 else end - 1
            window = min(self.savgol_window, maximum_window)
            if window % 2 == 0:
                window -= 1
            if window <= self.savgol_order:
                raise ValueError("not enough samples for the Savitzky-Golay window")

            savgol_signal = savgol_filter(
                y, window_length=window, polyorder=self.savgol_order
            )
            residual = y - savgol_signal
            centered_residual = residual - np.median(residual)
            noise_sigma = np.median(np.abs(centered_residual)) / 0.6745
            smoothing_factor = (
                end * noise_sigma**2 * self.spline_strength
            )

            weights = np.ones(end)
            candidate_peaks, _ = find_peaks(
                savgol_signal,
                prominence=self.candidate_prominence_fraction * np.ptp(savgol_signal),
                distance=window,
            )
            peak_radius = max(2, window // 3)
            for peak in candidate_peaks:
                left = max(0, peak - peak_radius)
                right = min(end, peak + peak_radius + 1)
                weights[left:right] = self.peak_weight

            spline = UnivariateSpline(
                x,
                savgol_signal,
                k=self.spline_order,
                s=smoothing_factor,
                w=weights,
            )
            smoothed_values[sample_index, :end] = spline(x)

        output = signal.copy(data=smoothed_values)
        output.attrs = dict(signal.attrs)
        output.attrs["description"] = (
            "Peak-preserving Savitzky-Golay and adaptive spline smoothing"
        )
        self.output = {self.output_variable: output}
        return self


class PeakFitPreprocessor(Preprocessor):
    """Detect spline peaks and fit them with lmfit pseudo-Voigt components.

    ``feature_variable`` supplies the smoothed curve used for initial peak
    locations, prominences, and widths. ``input_variable`` supplies the curve
    fitted by lmfit and may be either the original signal or another processed
    curve. Set ``feature_domain_variable`` when feature coordinates vary by
    sample. By default, fitted centers are bounded by midpoints between adjacent
    detected peaks; ``center_tolerance`` replaces those bounds with a fixed
    tolerance around each detected center.
    """

    def __init__(
        self,
        input_variable: str = "signal",
        domain_variable: str = "x",
        feature_variable: str = "spline_signal",
        feature_dim: str = "spline_x",
        feature_domain_variable: str | None = None,
        output_prefix: str = "fitted_peak",
        sample_dim: str = "sample",
        peak_dim: str = "detected_peak",
        minimum_prominence: float = 0.0,
        prominence_fraction: float = 0.025,
        peak_distance: int = 25,
        center_tolerance: float | None = None,
        name: str = "PeakFitPreprocessor",
    ) -> None:
        if any(
            not np.isfinite(value) or value < 0
            for value in (minimum_prominence, prominence_fraction)
        ):
            raise ValueError("prominence thresholds must be finite and non-negative")
        if not isinstance(peak_distance, int) or peak_distance < 1:
            raise ValueError("peak_distance must be a positive integer")
        if center_tolerance is not None and (
            not np.isfinite(center_tolerance) or center_tolerance <= 0
        ):
            raise ValueError("center_tolerance must be finite and positive or None")
        outputs = [
            f"{output_prefix}_{label}"
            for label in ("location", "prominence", "fwhm")
        ]
        inputs = [domain_variable, input_variable, feature_variable]
        if feature_domain_variable is not None and feature_domain_variable not in inputs:
            inputs.append(feature_domain_variable)
        super().__init__(
            name=name,
            input_variable=inputs,
            output_variable=outputs,
        )
        self.domain_variable = domain_variable
        self.feature_variable = feature_variable
        self.feature_dim = feature_dim
        self.feature_domain_variable = feature_domain_variable
        self.output_prefix = output_prefix
        self.sample_dim = sample_dim
        self.peak_dim = peak_dim
        self.minimum_prominence = float(minimum_prominence)
        self.prominence_fraction = float(prominence_fraction)
        self.peak_distance = peak_distance
        self.center_tolerance = (
            None if center_tolerance is None else float(center_tolerance)
        )

    def calculate(self, dataset: xr.Dataset) -> Self:
        domain, signal, _ = _curve_data(
            dataset, self.domain_variable, self.input_variable, self.sample_dim
        )
        if self.feature_variable not in dataset:
            raise ValueError(f"Dataset has no variable {self.feature_variable!r}")
        if self.feature_domain_variable is None:
            feature_curves = dataset[self.feature_variable]
            if self.feature_dim not in feature_curves.dims:
                raise ValueError(
                    f"{self.feature_variable!r} must have dimension {self.feature_dim!r}"
                )
            unexpected = set(feature_curves.dims).difference(
                {self.sample_dim, self.feature_dim}
            )
            if unexpected:
                raise ValueError(
                    f"{self.feature_variable!r} has unsupported dimensions "
                    f"{sorted(unexpected)}"
                )
            if self.sample_dim not in feature_curves.dims:
                feature_curves = feature_curves.expand_dims({self.sample_dim: [0]})
            feature_curves = feature_curves.transpose(self.sample_dim, self.feature_dim)
            shared_feature_x = np.asarray(
                feature_curves[self.feature_dim].values, dtype=float
            )
            if (
                shared_feature_x.ndim != 1
                or shared_feature_x.size < 3
                or not np.isfinite(shared_feature_x).all()
                or not np.all(np.diff(shared_feature_x) > 0)
            ):
                raise ValueError(
                    "feature coordinates must be finite and strictly increasing"
                )
            feature_domain = None
        else:
            feature_domain, feature_curves, _ = _curve_data(
                dataset,
                self.feature_domain_variable,
                self.feature_variable,
                self.sample_dim,
            )
            shared_feature_x = None
        if not np.array_equal(
            signal[self.sample_dim].values, feature_curves[self.sample_dim].values
        ):
            raise ValueError("fit and feature curves must have identical sample coordinates")

        detected_rows = []
        for sample_index, feature_y in enumerate(feature_curves.values):
            if feature_domain is None:
                feature_x = shared_feature_x
                feature_y = np.asarray(feature_y, dtype=float)
                if not np.isfinite(feature_y).all():
                    raise ValueError(
                        f"feature curve at sample {sample_index} contains non-finite values"
                    )
            else:
                _, (feature_x, feature_y) = _trim_curve(
                    feature_domain.values[sample_index], feature_y
                )
            indices, properties = find_peaks(
                feature_y,
                prominence=max(
                    self.minimum_prominence,
                    self.prominence_fraction * np.ptp(feature_y),
                ),
                distance=self.peak_distance,
            )
            if not len(indices):
                detected_rows.append(np.empty((0, 5)))
                continue
            widths = peak_widths(
                feature_y,
                indices,
                rel_height=0.5,
                prominence_data=(
                    properties["prominences"],
                    properties["left_bases"],
                    properties["right_bases"],
                ),
            )
            left = np.interp(widths[2], np.arange(feature_x.size), feature_x)
            right = np.interp(widths[3], np.arange(feature_x.size), feature_x)
            detected_rows.append(
                np.column_stack(
                    [
                        feature_x[indices],
                        properties["prominences"],
                        right - left,
                        left,
                        right,
                    ]
                )
            )

        peak_count = max((len(row) for row in detected_rows), default=0)
        fitted_features = np.full((len(detected_rows), peak_count, 3), np.nan)
        fitted_parameters = np.full_like(fitted_features, np.nan)
        fitted_values = np.full(signal.shape, np.nan)
        background_values = np.full(signal.shape, np.nan)
        fit_status = np.full(signal.sizes[self.sample_dim], "no_peaks", dtype="U8")
        fit_rmse = np.full(signal.sizes[self.sample_dim], np.nan)

        for sample_index, (x, y, detected_row) in enumerate(
            zip(domain.values, signal.values, detected_rows)
        ):
            end, (x, y) = _trim_curve(x, y)
            detected = detected_row[:, :3]
            left = detected_row[:, 3]
            right = detected_row[:, 4]
            try:
                result_features, parameters, fitted, background = self._fit_curve(
                    x, y, detected, left, right
                )
            except (ValueError, RuntimeError, ArithmeticError) as exc:
                sample = signal[self.sample_dim].values[sample_index]
                raise RuntimeError(f"lmfit fitting failed for sample {sample!r}: {exc}") from exc
            fitted_features[sample_index, :len(detected)] = result_features
            fitted_parameters[sample_index, :len(detected)] = parameters
            fitted_values[sample_index, :end] = fitted
            background_values[sample_index, :end] = background
            fit_rmse[sample_index] = np.sqrt(np.mean((y - fitted) ** 2))
            if len(detected):
                fit_status[sample_index] = "success"

        coords = {
            self.sample_dim: signal[self.sample_dim],
            self.peak_dim: np.arange(peak_count),
        }
        self.output = {}
        for index, label in enumerate(("location", "prominence", "fwhm")):
            self.output[f"{self.output_prefix}_{label}"] = xr.DataArray(
                fitted_features[:, :, index],
                dims=(self.sample_dim, self.peak_dim),
                coords=coords,
            )
        for index, label in enumerate(("amplitude", "sigma", "fraction")):
            self.output[f"{self.output_prefix}_{label}"] = xr.DataArray(
                fitted_parameters[:, :, index],
                dims=(self.sample_dim, self.peak_dim),
                coords=coords,
            )
        self.output["fitted_curve"] = signal.copy(data=fitted_values)
        self.output["fitted_background"] = signal.copy(data=background_values)
        self.output["fit_status"] = xr.DataArray(
            fit_status,
            dims=(self.sample_dim,),
            coords={self.sample_dim: signal[self.sample_dim]},
        )
        self.output["fit_rmse"] = xr.DataArray(
            fit_rmse,
            dims=(self.sample_dim,),
            coords={self.sample_dim: signal[self.sample_dim]},
        )
        feature_domain_units = (
            feature_curves[self.feature_dim].attrs.get("units")
            if feature_domain is None
            else feature_domain.attrs.get("units")
        )
        domain_units = domain.attrs.get("units") or feature_domain_units
        signal_units = signal.attrs.get("units") or feature_curves.attrs.get("units")
        for label in ("location", "fwhm", "sigma"):
            if domain_units:
                self.output[f"{self.output_prefix}_{label}"].attrs["units"] = domain_units
        if signal_units:
            self.output[f"{self.output_prefix}_prominence"].attrs["units"] = signal_units
            self.output["fit_rmse"].attrs["units"] = signal_units
        if signal_units and domain_units:
            self.output[f"{self.output_prefix}_amplitude"].attrs["units"] = (
                f"{signal_units} * {domain_units}"
            )
        self.output[f"{self.output_prefix}_fraction"].attrs["units"] = "1"
        return self

    def _fit_curve(self, x, y, detected, left, right):
        """Jointly fit positive peaks and a shifted exponential/linear background."""
        from lmfit.models import ExponentialModel, LinearModel, PseudoVoigtModel

        if not len(detected) or np.ptp(y) == 0:
            return detected.copy(), np.full((len(detected), 3), np.nan), y.copy(), y.copy()
        signal_scale = np.max(np.abs(y))
        scaled_y = y / signal_scale
        span = x[-1] - x[0]
        origin = x[0]

        class ShiftedExponentialModel(ExponentialModel):
            def eval(self, params=None, **kwargs):
                kwargs["x"] = (np.asarray(kwargs["x"]) - origin) / span
                return super().eval(params=params, **kwargs)

        exponential = ShiftedExponentialModel(prefix="exponential_")
        linear = LinearModel(prefix="linear_")
        model = exponential + linear
        params = model.make_params()
        mask = np.ones(x.size, dtype=bool)
        for lower, upper in zip(left, right):
            mask &= ~((x >= lower) & (x <= upper))
        if mask.sum() < 2:
            mask[:] = True
        slope, intercept = np.polyfit(x[mask] - origin, scaled_y[mask], 1)
        params["linear_slope"].set(value=slope)
        params["linear_intercept"].set(value=intercept - slope * origin)
        params["exponential_amplitude"].set(value=0)
        params["exponential_decay"].set(value=0.2, min=0.01, max=100)
        minimum_sigma = 0.5 * np.min(np.diff(x))
        maximum_sigma = span / 2
        centers = detected[:, 0]
        midpoints = (centers[:-1] + centers[1:]) / 2
        for index, (center, height, width) in enumerate(detected):
            prefix = f"peak{index}_"
            peak = PseudoVoigtModel(prefix=prefix)
            peak_params = peak.make_params()
            sigma = np.clip(width / 2, minimum_sigma, maximum_sigma)
            height_per_area = (
                0.5 * np.sqrt(np.log(2) / np.pi) + 0.5 / np.pi
            ) / sigma
            peak_params[prefix + "center"].set(
                value=center,
                min=(
                    (x[0] if index == 0 else midpoints[index - 1])
                    if self.center_tolerance is None
                    else max(x[0], center - self.center_tolerance)
                ),
                max=(
                    (x[-1] if index == len(detected) - 1 else midpoints[index])
                    if self.center_tolerance is None
                    else min(x[-1], center + self.center_tolerance)
                ),
            )
            peak_params[prefix + "sigma"].set(
                value=sigma, min=minimum_sigma, max=maximum_sigma
            )
            peak_params[prefix + "amplitude"].set(
                value=height / signal_scale / height_per_area, min=0
            )
            peak_params[prefix + "fraction"].set(value=0.5, min=0, max=1)
            params.update(peak_params)
            model = model + peak
        result = model.fit(
            scaled_y, params=params, x=x, method="least_squares", max_nfev=5000
        )
        if not result.success:
            raise RuntimeError(f"lmfit failed: {result.message}")
        fitted = result.eval(x=x) * signal_scale
        components = result.eval_components(x=x)
        background = (components["exponential_"] + components["linear_"]) * signal_scale
        fitted_features = np.empty((len(detected), 3))
        parameters = np.empty_like(fitted_features)
        for index in range(len(detected)):
            prefix = f"peak{index}_"
            fitted_features[index] = (
                result.params[prefix + "center"].value,
                result.params[prefix + "height"].value * signal_scale,
                result.params[prefix + "fwhm"].value,
            )
            parameters[index] = (
                result.params[prefix + "amplitude"].value * signal_scale,
                result.params[prefix + "sigma"].value,
                result.params[prefix + "fraction"].value,
            )
        arrays = (fitted, background, fitted_features, parameters)
        if not all(np.isfinite(values).all() for values in arrays):
            raise ValueError("lmfit produced non-finite fit results")
        return fitted_features, parameters, fitted, background


class UnitCubeTransform(Preprocessor):
    """Transform component-valued data between physical and unit-cube space.

    The component coordinate is used to align each column with its declared
    bound, avoiding any dependence on dictionary insertion order.  Set
    ``inverse=True`` to map unit-cube values back to physical coordinates.
    """

    def __init__(
        self,
        bounds: dict[str, dict[str, float]],
        input_variable: str,
        output_variable: str,
        component_dim: str = "component",
        inverse: bool = False,
        name: str = "UnitCubeTransform",
    ) -> None:
        if not bounds:
            raise ValueError("bounds must not be empty")
        normalized_bounds: dict[str, dict[str, float]] = {}
        for component, limit in bounds.items():
            if "min" not in limit or "max" not in limit:
                raise ValueError(
                    f"bounds for {component!r} must contain 'min' and 'max'"
                )
            lower, upper = float(limit["min"]), float(limit["max"])
            if not np.isfinite([lower, upper]).all() or lower >= upper:
                raise ValueError(
                    f"bounds for {component!r} must be finite with min < max"
                )
            normalized_bounds[component] = {"min": lower, "max": upper}

        super().__init__(
            name=name,
            input_variable=input_variable,
            output_variable=output_variable,
        )
        self.bounds = normalized_bounds
        self.component_dim = component_dim
        self.inverse = bool(inverse)

    def calculate(self, dataset: xr.Dataset) -> Self:
        values = self._get_variable(dataset)
        if self.component_dim not in values.dims:
            raise ValueError(
                f"{self.input_variable!r} must have dimension {self.component_dim!r}"
            )
        if self.component_dim not in values.coords:
            raise ValueError(
                f"{self.input_variable!r} must have component-name coordinates"
            )

        components = list(values[self.component_dim].values)
        missing = [
            str(component) for component in components if component not in self.bounds
        ]
        if missing:
            raise ValueError(f"bounds are missing components: {missing}")
        lower = xr.DataArray(
            [self.bounds[component]["min"] for component in components],
            dims=(self.component_dim,),
            coords={self.component_dim: components},
        )
        span = xr.DataArray(
            [
                self.bounds[component]["max"] - self.bounds[component]["min"]
                for component in components
            ],
            dims=(self.component_dim,),
            coords={self.component_dim: components},
        )
        transformed = (
            lower + values * span if self.inverse else (values - lower) / span
        )
        transformed = transformed.transpose(*values.dims)
        transformed.attrs = dict(values.attrs)
        transformed.attrs.update(
            {
                "coordinate_space": "physical" if self.inverse else "unit_cube",
                "transformed_from": self.input_variable,
            }
        )
        self.output = {self.output_variable: transformed}
        return self


class SplinePreprocessor(Preprocessor):
    """Resample one-dimensional spectra onto a common spline grid.

    Defaults are appropriate for UV-Vis spectra: cubic interpolation on 201
    evenly spaced points with no additional smoothing. Set ``smoothing_factor``
    to a positive value for noisy measurements. The output uses a separate
    dimension name so it can be merged with data on the original grid.

    Parameters
    ----------
    input_variable : str, default="spectrum"
        Variable containing one spectrum or a batch of spectra.
    output_variable : str, default="spline_spectrum"
        Variable in which to store the resampled spectra.
    dim : str, default="wavelength"
        Coordinate along the spectral domain.
    output_dim : str, default="spline_wavelength"
        Dimension name for the common output grid.
    sample_dim : str, default="sample"
        Optional batch dimension.
    n_points : int, default=201
        Number of evenly spaced output points.
    spline_degree : int, default=3
        Spline degree accepted by :class:`scipy.interpolate.UnivariateSpline`.
    smoothing_factor : float or None, default=0.0
        Spline smoothing parameter. Zero gives interpolation.
    x_min, x_max : float or None
        Output range. By default the full input domain is used.
    """

    def __init__(
        self,
        input_variable: str = "spectrum",
        output_variable: str = "spline_spectrum",
        dim: str = "wavelength",
        output_dim: str = "spline_wavelength",
        sample_dim: str = "sample",
        n_points: int = 201,
        spline_degree: int = 3,
        smoothing_factor: float | None = 0.0,
        x_min: float | None = None,
        x_max: float | None = None,
        name: str = "SplinePreprocessor",
    ) -> None:
        super().__init__(
            name=name, input_variable=input_variable, output_variable=output_variable
        )
        if n_points < 2:
            raise ValueError("n_points must be at least 2")
        if not 1 <= spline_degree <= 5:
            raise ValueError("spline_degree must be between 1 and 5")
        if smoothing_factor is not None and smoothing_factor < 0:
            raise ValueError("smoothing_factor must be non-negative or None")
        if x_min is not None and x_max is not None and x_min >= x_max:
            raise ValueError("x_min must be less than x_max")
        if x_min is not None and not np.isfinite(x_min):
            raise ValueError("x_min must be finite")
        if x_max is not None and not np.isfinite(x_max):
            raise ValueError("x_max must be finite")
        self.dim = dim
        self.output_dim = output_dim
        self.sample_dim = sample_dim
        self.n_points = n_points
        self.spline_degree = spline_degree
        self.smoothing_factor = smoothing_factor
        self.x_min = x_min
        self.x_max = x_max

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Spline-resample the configured variable in ``dataset``."""
        data = self._get_variable(dataset)
        if self.dim not in data.dims:
            raise ValueError(
                f"Input variable {self.input_variable!r} must include dim={self.dim!r}"
            )
        unexpected_dims = set(data.dims) - {self.sample_dim, self.dim}
        if unexpected_dims:
            raise ValueError(
                "Expected 1D spectra with an optional sample dimension; "
                f"unexpected dimensions: {sorted(unexpected_dims)}"
            )

        has_sample_dim = self.sample_dim in data.dims
        if not has_sample_dim:
            data = data.expand_dims({self.sample_dim: [0]})
        data = data.transpose(self.sample_dim, self.dim)
        domain = np.asarray(data.coords[self.dim].values, dtype=float)
        if domain.ndim != 1 or not np.isfinite(domain).all():
            raise ValueError(
                f"Coordinate {self.dim!r} must be finite and one-dimensional"
            )

        order = np.argsort(domain)
        domain = domain[order]
        values = np.asarray(data.values, dtype=float)[:, order]
        unique_domain, inverse = np.unique(domain, return_inverse=True)
        if unique_domain.size <= self.spline_degree:
            raise ValueError(
                f"At least {self.spline_degree + 1} unique points are required"
            )

        x_min = unique_domain[0] if self.x_min is None else float(self.x_min)
        x_max = unique_domain[-1] if self.x_max is None else float(self.x_max)
        if x_min < unique_domain[0] or x_max > unique_domain[-1]:
            raise ValueError(
                "Requested spline range lies outside the measured domain "
                f"[{unique_domain[0]}, {unique_domain[-1]}]"
            )
        output_domain = np.linspace(x_min, x_max, self.n_points)

        resampled = np.empty((values.shape[0], self.n_points), dtype=float)
        for sample_index, spectrum in enumerate(values):
            sums = np.zeros(unique_domain.size, dtype=float)
            counts = np.zeros(unique_domain.size, dtype=int)
            finite = np.isfinite(spectrum)
            np.add.at(sums, inverse[finite], spectrum[finite])
            np.add.at(counts, inverse[finite], 1)
            valid = counts > 0
            if valid.sum() <= self.spline_degree:
                raise ValueError(
                    f"Sample {sample_index} has fewer than {self.spline_degree + 1} "
                    "finite, unique spectral points"
                )
            sample_domain = unique_domain[valid]
            sample_values = sums[valid] / counts[valid]
            if (
                output_domain[0] < sample_domain[0]
                or output_domain[-1] > sample_domain[-1]
            ):
                raise ValueError(
                    f"Sample {sample_index} does not cover the requested spline range"
                )
            spline = UnivariateSpline(
                sample_domain,
                sample_values,
                k=self.spline_degree,
                s=self.smoothing_factor,
                ext="raise",
            )
            resampled[sample_index] = spline(output_domain)

        output = xr.DataArray(
            resampled,
            dims=(self.sample_dim, self.output_dim),
            coords={
                self.sample_dim: data.coords[self.sample_dim].values,
                self.output_dim: output_domain,
            },
            attrs={
                "description": "Spectrum resampled with scipy.interpolate.UnivariateSpline",
                "source_dimension": self.dim,
            },
        )
        if not has_sample_dim:
            output = output.isel({self.sample_dim: 0}, drop=True)
        self.output[self.output_variable] = output
        return self

class LogLogTransform(Preprocessor):
    """Pre-processing class to transform SAXS data into log-log scale.

    Parameters
    ----------
    input_variable : str
        The name of the `xarray.Dataset` data variable to extract from the input dataset
    output_variable : str
        The name of the variable to be inserted into the `xarray.Dataset` by this `PipelineOp`
    dim: str
        The dimension in the `xarray.Dataset` to apply this transform over
    name : str
        The name to use when added to a Pipeline. This name is used when calling Pipeline.search()
    """    
    def __init__(
        self,
        input_variable: str = None,
        output_variable: str = None,
        dim: str = "q",
        name: str = "PreprocessorBase",
    ) -> None:
        super().__init__(
            name=name, input_variable=input_variable, output_variable=output_variable
        )
        self.dim = dim

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this `PipelineOp` to the supplied `xarray.Dataset`."""
        data = self._get_variable(dataset)

        domain_variable = "log_" + self.dim
        domain_values = np.log10(data[self.dim].values)
        codomain_values = np.log10(np.abs(data.values))
    
        coords = {}
        for d in data.dims:
            if d!=self.dim:
                coords[d] = data.coords[d]
            else:
                coords[domain_variable] = domain_values
        
        dims = tuple(d if d != self.dim else domain_variable for d in data.dims)

        # Create the output variable with correct dims and coords
        self.output[self.output_variable] = xr.DataArray(
            codomain_values,
            dims=dims,
            coords=coords,
            attrs={"description": "Co-domain log transformed"}
        )

        return self

    
class SavgolFilter(Preprocessor):
    """Smooth and take derivatives of input data via a Savitsky-Golay filter

    This `PipelineOp` cleans measurement data and takes smoothed derivatives using `scipy.signal.savgol_filter`. Below
    is a summary of the steps taken.

    Parameters
    ----------
    input_variable : str
        The name of the `xarray.Dataset` data variable to extract from the input dataset
    output_variable : str
        The name of the variable to be inserted into the `xarray.Dataset` by this `PipelineOp`
    dim: str
        The dimension in the `xarray.Dataset` to apply this filter over
    xlo, xhi: Optional[Number]
        The values of the input dimension (dim, above) to trim the data to
    xlo_isel, xhi_isel: Optional[int]
        The integer indices of the input dimension (dim, above) to trim the data to
    pedestal: Optional[Number]
        This value is added to the input_variable to establish a fixed data 'floor'
    npts: int
        The size of the grid to interpolate onto
    derivative: int
        The order of the derivative to return. If derivative=0, the data is smoothed with no derivative taken.
    window_length: int
        The width of the window used in the savgol smoothing. See `scipy.signal.savgol_filter` for more information.
    polyorder: int
        The order of polynomial used in the savgol smoothing. See `scipy.signal.savgol_filter` for more information.
    apply_log_scale: bool
        If True, the `input_variable` and associated `dim` coordinated are scaled with `numpy.log10`
    name: str
        The name to use when added to a Pipeline. This name is used when calling Pipeline.search()

    Notes
    -----
    This `PipelineOp` performs the following steps:

    1. Data is trimmed to `(xlo, xhi)` and then `(xlo_isel, xhi_isel)` in that order. The former trims the data to a
    numerical while the latter trims to integer indices. It is generally not advisable to supply both varieties and a
    warning will be raised if this is attempted.

    2. If `apply_log_scale = True`, both the `input_variable` and `dim` data will be scaled with `numpy.log10`. A new
    `xarray` dimension and coordinate will be created with the name `log_{dim}`.

    3. All duplicate data (multiple data values at the same `dim` coordinates) are removed by taking the average of the
    duplicates.

    4. If `pedestal` is specified, the pedestal value is added to the data and all NaNs are filled with the pedestal

    5. The data is interpolated onto a constant grid with `npts` values from the trimmed minimum to the trimmed maximum.  If `apply_log_scale=True`, the grid is geometrically rather than linearly spaced.

    6. All remaining NaN values are dropped along `dim`

    7. Finally, `scipy.signal.savgol_filter` is applied with the `window_length`, `polyorder`, and `derivative`
    parameters specified in the constructor.

    """

    def __init__(
        self,
        input_variable: str,
        output_variable: str,
        dim: str = "q",
        xlo: Optional[Number] = None,
        xhi: Optional[Number] = None,
        xlo_isel: Optional[int] = None,
        xhi_isel: Optional[int] = None,
        pedestal: Optional[Number] = None,
        npts: int = 250,
        derivative: int = 0,
        window_length: int = 31,
        polyorder: int = 2,
        apply_log_scale: bool = True,
        name: str = "SavgolFilter",
    ) -> None:
        super().__init__(name=name, input_variable=input_variable, output_variable=output_variable)

        self.dim = dim
        self.npts = npts
        self.xlo = xlo
        self.xhi = xhi
        self.xlo_isel = xlo_isel
        self.xhi_isel = xhi_isel
        self.pedestal = pedestal
        self.derivative = derivative
        self.apply_log_scale = apply_log_scale
        self.polyorder = polyorder
        self.window_length = window_length

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this `PipelineOp` to the supplied `xarray.dataset`"""
        data1 = self._get_variable(dataset)

        if self.dim not in data1.coords:
            raise ValueError(
                (
                    f"""dim argument of PipelineOp '{self.name}' set to '{self.dim}' but not found in coords"""
                )
            )

        if (self.xlo is not None) or (self.xhi is not None):
            if (self.xlo_isel is not None) or (self.xhi_isel is not None):
                warnings.warn(
                    (
                        "Both value and index based indexing have been specified! "
                        "Value indexing will be applied first!"
                    ),
                    stacklevel=2,
                )

        data1 = data1.transpose(self.dim, ...)

        data1 = data1.sel({self.dim: slice(self.xlo, self.xhi)})
        data1 = data1.isel({self.dim: slice(self.xlo_isel, self.xhi_isel)})

        if self.apply_log_scale:
            dim = "log_" + self.dim
            data1 = data1.where(data1 > 0.0, drop=True)
            data1 = data1.pipe(np.log10)
            data1[self.dim] = np.log10(data1[self.dim])
            data1 = data1.rename({self.dim: dim})
        else:
            dim = "sg_" + self.dim
            data1[dim] = data1[self.dim]

        # need to remove duplicate values
        data1 = data1.groupby(dim, squeeze=False).mean()

        # set minimum value of scattering to pedestal value and fill nans with this value
        if self.pedestal is not None:
            data1 += self.pedestal
            data1 = data1.where(~np.isnan(data1)).fillna(self.pedestal)

        # interpolate to constant lin or log(dq) grid
        x_new = np.linspace(data1[dim].min().item(), data1[dim].max().item(), self.npts)
        dx = float(x_new[1] - x_new[0])
        data1 = data1.bfill(dim).ffill(dim).interpolate_na(dim)
        data1 = data1.interp({dim: x_new})

        # filter out any q that have NaN
        data1 = data1.dropna(dim, how="any")

        # take derivative
        data1_filtered = savgol_filter(
            data1.values,
            window_length=self.window_length,
            polyorder=self.polyorder,
            delta=dx,
            axis=0,
            deriv=self.derivative,
        )

        self.output[self.output_variable] = data1.copy(data=data1_filtered).transpose(..., dim)
        self.output[self.output_variable].attrs["description"] = f"Savitsky-Golay filtered data"

        return self


class SubtractMin(Preprocessor):
    """Baseline input variable by subtracting minimum value"""

    def __init__(
        self,
        input_variable: str,
        output_variable: str,
        dim: str,
        name: str = "SubtractMin",
    ) -> None:
        """
        Parameters
        ----------
        input_variable : str
            The name of the `xarray.Dataset` data variable to extract from the input dataset

        output_variable : str
            The name of the variable to be inserted into the `xarray.Dataset` by this `PipelineOp`

        dim: str
             The dimension to use when calculating the data minimum

        name: str
            The name to use when added to a Pipeline. This name is used when calling Pipeline.search(
        """
        super().__init__(name=name, input_variable=input_variable, output_variable=output_variable)

        self.dim = dim

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this `PipelineOp` to the supplied `xarray.dataset`"""
        data1 = self._get_variable(dataset)

        self.output[self.output_variable] = data1 - data1.min(self.dim)
        self.output[self.output_variable].attrs[
            "description"
        ] = f"Subtracted minimum value along dimension '{self.dim}'"

        return self


class Subtract(Preprocessor):
    """Baseline input variable by subtracting a value"""

    def __init__(
        self,
        input_variable: str,
        output_variable: str,
        dim: str,
        value: float | str,
        coord_value: bool = True,
        name: str = "Subtract",
    ) -> None:
        """
        Parameters
        ----------
        input_variable : str
            The name of the `xarray.Dataset` data variable to extract from the input dataset

        output_variable : str
            The name of the variable to be inserted into the `xarray.Dataset` by this `PipelineOp`

        dim: str
             The dimension to use when calculating the data minimum

        name: str
            The name to use when added to a Pipeline. This name is used when calling Pipeline.search(
        """
        super().__init__(name=name, input_variable=input_variable, output_variable=output_variable)

        self.dim = dim
        self.value = value
        self.coord_value = coord_value

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this `PipelineOp` to the supplied `xarray.dataset`"""
        data1 = self._get_variable(dataset)

        if isinstance(self.value, str) and self.coord_value:
            value = dataset[self.value].item()
            subtract_value = data1.sel({self.dim: value}, method="nearest")
        elif self.coord_value:
            subtract_value = data1.sel({self.dim: self.value}, method="nearest")
        else:
            subtract_value = self.value

        self.output[self.output_variable] = data1 - subtract_value
        self.output[self.output_variable].attrs[
            "description"
        ] = f"Subtracted minimum value along dimension '{self.dim}'"

        return self


class Standardize(Preprocessor):
    """Standardize the data to have min 0 and max 1

    Parameters
    ----------
    input_variable : str
        The name of the `xarray.Dataset` data variable to extract from the input dataset
    output_variable : str
        The name of the variable to be inserted into the `xarray.Dataset` by this `PipelineOp`
    dim : str
        The dimension used for calculating the data minimum
    component_dim : str | None, default="component"
        The dimension for component-wise operations
    scale_variable : str | None, default=None
        If specified, the min/max of this data variable in the supplied `xarray.Dataset` will be used to scale the
        data rather than min/max of the `input_variable` or the supplied `min_val` or `max_val`
    min_val : dict | Number | None, default=None
        Value used to scale the data minimum
    max_val : dict | Number | None, default=None
        Value used to scale the data maximum
    name : str, default="Standardize"
        The name to use when added to a Pipeline
    """

    def __init__(
        self,
        input_variable: str,
        output_variable: str,
        dim: str,
        component_dim: str | None = "component",
        scale_variable: str | None = None,
        min_val: dict | Number | None = None,
        max_val: dict | Number | None = None,
        name: str = "Standardize",
    ) -> None:
        super().__init__(name=name, input_variable=input_variable, output_variable=output_variable)

        self.dim = dim
        self.component_dim = component_dim
        self.min_val = min_val
        self.max_val = max_val
        self.scale_variable = scale_variable

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this `PipelineOp` to the supplied `xarray.dataset`"""
        data1 = self._get_variable(dataset)

        if self.max_val is None:
            if self.scale_variable is None:
                max_val = data1.max(self.dim)
            else:
                max_val = dataset[self.scale_variable].max(self.dim)
        elif isinstance(self.max_val, dict):
            max_val = xr.Dataset(self.max_val).to_array(self.component_dim)
        else:
            max_val = self.max_val

        if self.min_val is None:
            if self.scale_variable is None:
                min_val = data1.min(self.dim)
            else:
                min_val = dataset[self.scale_variable].min(self.dim)
        elif isinstance(self.min_val, dict):
            min_val = xr.Dataset(self.min_val).to_array(self.component_dim)
        else:
            min_val = self.min_val

        self.output[self.output_variable] = (data1 - min_val) / (max_val - min_val)
        self.output[self.output_variable].attrs[
            "description"
        ] = f"Data normalized to have range 0 -> 1"

        return self


class Destandardize(Preprocessor):
    """Transform the data from 0->1 scaling

    Parameters
    ----------
    input_variable : str
        The name of the `xarray.Dataset` data variable to extract from the input dataset
    output_variable : str
        The name of the variable to be inserted into the `xarray.Dataset` by this `PipelineOp`
    dim : str
        The dimension used for calculating the data minimum
    component_dim : str | None, default="component"
        The dimension for component-wise operations
    scale_variable : str | None, default=None
        If specified, the min/max of this data variable in the supplied `xarray.Dataset` will be used to scale the
        data rather than min/max of the `input_variable` or the supplied `min_val` or `max_val`
    min_val : Number | None, default=None
        Value used to scale the data minimum
    max_val : Number | None, default=None
        Value used to scale the data maximum
    name : str, default="Destandardize"
        The name to use when added to a Pipeline
    """

    def __init__(
        self,
        input_variable: str,
        output_variable: str,
        dim: str,
        component_dim: str | None = "component",
        scale_variable: str | None = None,
        min_val: dict | Number | None = None,
        max_val: dict | Number | None = None,
        name: str = "Destandardize",
    ) -> None:

        super().__init__(name=name, input_variable=input_variable, output_variable=output_variable)

        self.dim = dim
        self.component_dim = component_dim
        self.min_val = min_val
        self.max_val = max_val
        self.scale_variable = scale_variable

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this `PipelineOp` to the supplied `xarray.dataset`"""
        data1 = self._get_variable(dataset)

        if self.max_val is None:
            if self.scale_variable is None:
                max_val = data1.max(self.dim)
            else:
                max_val = dataset[self.scale_variable].max(self.dim)
        elif isinstance(self.max_val, dict):
            max_val = xr.Dataset(self.max_val).to_array(self.component_dim)
        else:
            max_val = self.max_val

        if self.min_val is None:
            if self.scale_variable is None:
                min_val = data1.min(self.dim)
            else:
                min_val = dataset[self.scale_variable].min(self.dim)
        elif isinstance(self.min_val, dict):
            min_val = xr.Dataset(self.min_val).to_array(self.component_dim)
        else:
            min_val = self.min_val

        self.output[self.output_variable] = data1 * (max_val - min_val) + min_val
        self.output[self.output_variable].attrs[
            "description"
        ] = f"Data normalized to have range 0 -> 1"

        return self


class Zscale(Preprocessor):
    """Z-scale the data to have mean 0 and standard deviation scaling

    Parameters
    ----------
    input_variable : str
        The name of the `xarray.Dataset` data variable to extract from the input dataset

    output_variable : str
        The name of the variable to be inserted into the `xarray.Dataset` by this `PipelineOp`

    dim: str
         The dimension to use when calculating the data minimum

    name: str
        The name to use when added to a Pipeline. This name is used when calling Pipeline.search(
    """

    def __init__(
        self, input_variable: str, output_variable: str, dim: str, name: str = "Zscale"
    ) -> None:
        super().__init__(name=name, input_variable=input_variable, output_variable=output_variable)

        self.mean = None
        self.std = None
        self.dim = dim

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this `PipelineOp` to the supplied `xarray.dataset`"""
        data1 = self._get_variable(dataset)
        self.mean = data1.mean()
        self.std = data1.std()

        # this standardizes between the min and the max values, this isn't z-scaling...
        self.output[self.output_variable] = (data1 - self.mean) / self.std
        self.output[self.output_variable].attrs[
            "description"
        ] = f"Data Zscaled to have range 0 ~ -1 -> 1"

        return self


class ZscaleError(Preprocessor):
    """Scale the y_err data, first input is y, second input is y_err

    Parameters
    ----------
    input_variables : Union[Optional[str], List[str]]
        The names of the input variables - first is y, second is y_err
    output_variable : str
        The name of the variable to be inserted into the dataset
    dim : str
        The dimension to use when calculating the data minimum
    name : str, default="Zscale_error"
        The name to use when added to a Pipeline
    """

    def __init__(
        self,
        input_variables: Union[Optional[str], List[str]],
        output_variable: str,
        dim: str,
        name: str = "Zscale_error",
    ) -> None:
        super().__init__(name=name, input_variable=input_variables, output_variable=output_variable)

        self.dim = dim

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this `PipelineOp` to the supplied `xarray.dataset`"""
        data1 = self._get_variable(dataset)
        data_y_err = data1[self.input_variable[1]]
        data_y = data1[self.input_variable[0]]

        self.output[self.output_variable] = data_y_err / data_y.std()
        self.output[self.output_variable].attrs[
            "description"
        ] = f"Uncertainty normalized to have range 0 -> inf"

        return self


class BarycentricToTernaryXY(Preprocessor):
    """
    Transform from ternary coordinates to xy coordinates

    Note ---- Adapted from BaryCentric transform mpltern:
    https://github.com/yuzie007/mpltern/blob/master/mpltern/ternary/transforms.py

    Parameters
    ----------
    input_variable : str
        The name of the `xarray.Dataset` data variable to extract from the input dataset

    output_variable : str
        The name of the variable to be inserted into the `xarray.Dataset` by this `PipelineOp`

    sample_dim: str
         The dimension to use when calculating the data minimum

    name: str
        The name to use when added to a Pipeline. This name is used when calling Pipeline.search(
    """

    def __init__(
        self,
        input_variable: str,
        output_variable: str,
        sample_dim: str,
        name: str = "BarycentricToTernaryXY",
    ) -> None:
        super().__init__(name=name, input_variable=input_variable, output_variable=output_variable)

        x_min = 0.5 - 1.0 / np.sqrt(3)
        x_max = 0.5 + 1.0 / np.sqrt(3)
        self.corners = np.array([(0.5, 1.0), (x_min, 0.0), (x_max, 0.0)])

        self.sample_dim = sample_dim

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this `PipelineOp` to the supplied `xarray.dataset`"""
        bary = self._get_variable(dataset)
        bary = bary.transpose(self.sample_dim, ...).values
        bary = bary / np.sum(bary, axis=1)[:, np.newaxis]
        xy = np.dot(bary, self.corners)

        self.output[self.output_variable] = xr.DataArray(xy, dims=[self.sample_dim, "xy"])
        self.output[self.output_variable].attrs["description"] = (
            "barycentric coordinates from xy variable " f"'{self.input_variable}'"
        )
        return self


class TernaryXYToBarycentric(Preprocessor):
    """Transform to ternary coordinates from xy coordinates

    Note
    ----
    Adapted from BaryCentric transform mpltern:
    https://github.com/yuzie007/mpltern/blob/master/mpltern/ternary/transforms.py

    Parameters
    ----------
    input_variable : str
        The name of the `xarray.Dataset` data variable to extract from the input dataset
    output_variable : str
        The name of the variable to be inserted into the `xarray.Dataset` by this `PipelineOp`
    sample_dim : str
        The dimension to use when calculating the data minimum
    name : str, default="TernaryXYToBarycentric"
        The name to use when added to a Pipeline
    """

    def __init__(
        self,
        input_variable: str,
        output_variable: str,
        sample_dim: str,
        name: str = "TernaryXYToBarycentric",
    ) -> None:
        """
        Parameters
        ----------
        input_variable : str
            The name of the `xarray.Dataset` data variable to extract from the input dataset

        output_variable : str
            The name of the variable to be inserted into the `xarray.Dataset` by this `PipelineOp`

        sample_dim: str
             The dimension to use when calculating the data minimum

        name: str
            The name to use when added to a Pipeline. This name is used when calling Pipeline.search(
        """
        super().__init__(name=name, input_variable=input_variable, output_variable=output_variable)

        x_min = 0.5 - 1.0 / np.sqrt(3)
        x_max = 0.5 + 1.0 / np.sqrt(3)
        self.corners = [(0.5, 1.0), (x_min, 0.0), (x_max, 0.0)]

        self.sample_dim = sample_dim

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this `PipelineOp` to the supplied `xarray.dataset`"""
        xy = self._get_variable(dataset)
        xy = xy.transpose(self.sample_dim, ...).values
        xys = np.column_stack((xy, np.ones(xy.shape[0])))
        v = np.column_stack((self.corners, np.ones(3)))
        bary = np.dot(xys, np.linalg.inv(v))
        self.output[self.output_variable] = xr.DataArray(bary, dims=[self.sample_dim, "component"])
        self.output[self.output_variable].attrs["description"] = (
            "XY coordinates from barycentric variable " f"'{self.input_variable}'"
        )
        return self


class SympyTransform(Preprocessor):
    """Transform data using sympy expressions

    Parameters
    ----------
    input_variable : str
        The name of the `xarray.Dataset` data variable to extract from the input dataset
    output_variable : str
        The name of the variable to be inserted into the `xarray.Dataset` by this `PipelineOp`
    sample_dim : str
        The sample dimension i.e., the dimension of compositions or grid points
    component_dim : str, default="component"
        The dimension of the component of each gridpoint
    transforms : dict[str,object]
        A dictionary of transforms (sympy expressions) to evaluate to generate new variables. For this method to
        function, the transforms must be completely specified except for the names in component_dim of the
        input_variable
    transform_dim : str
        The name of the dimension that the 'component_dim' will be transformed to
    name : str, default="SympyTransform"
        The name to use when added to a Pipeline

    Example
    -------
    ```python
    from AFL.double_agent import *
    import sympy
    with Pipeline() as p:
        CartesianGrid(
            output_variable='comps',
            grid_spec={
                'A':{'min':1,'max':25,'steps':5},
                'B':{'min':1,'max':25,'steps':5},
                'C':{'min':1,'max':25,'steps':5},
            },
            sample_dim='grid'
        )

        A,B,C = sympy.symbols('A B C')
        vA = A/(A+B+C)
        vB = B/(A+B+C)
        vC = C/(A+B+C)
        SympyTransform(
            input_variable='comps',
            output_variable='trans_comps',
            sample_dim='grid',
            transforms={'vA':vA,'vB':vB,'vC':vC},
            transform_dim='trans_component'
        )

    p.calculate(xr.Dataset())# returns dataset with grid and transformed grid
    ```
    """

    def __init__(
        self,
        input_variable: str,
        output_variable: str,
        sample_dim: str,
        transforms: dict[str, object],
        transform_dim: str,
        component_dim: str = "component",
        name: str = "SympyTransform",
    ) -> None:

        # must convert to strings for JSON serialization
        transforms = {k:str(v) for k,v in transforms.items()}

        super().__init__(
            name=name, input_variable=input_variable, output_variable=output_variable
        )
        self.sample_dim = sample_dim
        self.component_dim = component_dim
        self.transforms = transforms
        self.transform_dim = transform_dim

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this `PipelineOp` to the supplied `xarray.dataset`"""
        data = dataset[self.input_variable].transpose(self.sample_dim, self.component_dim)
        n = data.sizes[self.sample_dim]

        # need to construct a dict of arrays
        comps = {k: v.squeeze().values for k, v in data.groupby(self.component_dim, squeeze=False)}


        # apply transform
        new_comps = xr.Dataset()
        for name, transform in self.transforms.items():
            transform = sympy.sympify(transform)
            symbols = list(transform.free_symbols)
            lam = sympy.lambdify(symbols, transform)

            if symbols:
                result = lam(**{k.name: comps[k.name] for k in symbols})
            else:
                result = transform.evalf()

            if np.isscalar(result):
                result = np.full(n, float(result))

            new_comps[name] = ((self.sample_dim,), listify(result))

        new_comps = new_comps.to_array(self.transform_dim).transpose(..., self.transform_dim)

        self.output[self.output_variable] = new_comps
        self.output[self.output_variable].attrs[
            "description"
        ] = "Variables transformed using sympy expressions"

        return self


class Extrema(Preprocessor):
    """Find the extrema of a data variable"""

    def __init__(
        self,
        input_variable: str,
        output_variable: str,
        dim: str,
        return_coords: bool = False,
        operator="max",
        slice: Optional[List] = None,
        slice_dim: Optional[str] = None,
        name: str = "Extrema",
    ) -> None:
        """
        Parameters
        ----------
        input_variable : str
            The name of the `xarray.Dataset` data variable to extract from the input dataset

        output_variable : str
            The name of the variable to be inserted into the `xarray.Dataset` by this `PipelineOp`

        dim: str
             The dimension to use when calculating the data minimum

        name: str
            The name to use when added to a Pipeline. This name is used when calling Pipeline.search(
        """
        super().__init__(name=name, input_variable=input_variable, output_variable=output_variable)

        self.dim = dim
        self.operator = operator
        self.return_coords = return_coords
        self.slice = slice
        self.slice_dim = slice_dim

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Apply this `PipelineOp` to the supplied `xarray.dataset`"""
        data = self._get_variable(dataset)

        if self.slice is not None:
            data = data.sel({self.slice_dim: slice(*self.slice)})

        if self.return_coords:
            if "arg" not in self.operator:
                operator = "arg" + self.operator
            else:
                operator = self.operator
            idx = getattr(data, operator)(dim=self.dim)
            self.output[self.output_variable] = data[self.dim][idx]
        else:
            self.output[self.output_variable] = getattr(data, self.operator)(dim=self.dim)

        self.output[self.output_variable].attrs[
            "description"
        ] = f"Extrema of {self.input_variable} with operator {self.operator} and return_coords {self.return_coords}"

        return self


class VarsToArray(Preprocessor):
    """Convert multiple variables into a single array

    Parameters
    ----------
    input_variables : List
        List of input variables to combine into an array
    output_variable : str
        The name of the variable to be inserted into the dataset
    variable_dim : str
        The dimension name for the variables in the output array
    squeeze : bool, default=False
        Whether to squeeze out single-dimension axes
    variable_mapping : dict, default=None
        Optional mapping to rename variables
    name : str, default='VarsToArray'
        The name to use when added to a Pipeline
    """

    def __init__(
        self,
        input_variables: List,
        output_variable: str,
        variable_dim: str,
        squeeze: bool = False,
        variable_mapping: dict = None,
        name: str = "VarsToArray",
    ):

        super().__init__(name=name, input_variable=input_variables, output_variable=output_variable)
        print(self.input_variable, self.output_variable)

        if variable_mapping is None:
            self.variable_mapping = {}
        else:
            self.variable_mapping = variable_mapping

        self.variable_dim = variable_dim
        self.squeeze = squeeze

    def calculate(self, dataset):
        output = (
            dataset[self.input_variable]
            .rename_vars(self.variable_mapping)
            .to_array(self.variable_dim)
        )
        if self.squeeze:
            output = output.squeeze()
        print(self.output_variable)
        print(output)
        self.output[self.output_variable] = output
        return self


class ArrayToVars(Preprocessor):
    """Convert an array into multiple variables

    Parameters
    ----------
    input_variable : str
        The name of the array variable to split into separate variables
    output_variables : list
        The names of the variables to create from the array
    split_dim : str
        The dimension to split the array along
    postfix : str, default=''
        String to append to output variable names
    squeeze : bool, default=False
        Whether to squeeze out single-dimension axes
    name : str, default='DatasetToVars'
        The name to use when added to a Pipeline
    """

    def __init__(
        self,
        input_variable: str,
        output_variables: list,
        split_dim: list,
        postfix: str = "",
        squeeze: bool = False,
        name: str = "DatasetToVars",
    ):

        super().__init__(name=name, input_variable=input_variable, output_variable=output_variables)
        print(self.input_variable, self.output_variable)

        self.split_dim = split_dim
        self.squeeze = squeeze
        self.output_variables = output_variables
        self.postfix = postfix

    def calculate(self, dataset):
        input_arr = dataset[self.input_variable]
        if self.squeeze:
            input_arr = input_arr.squeeze()
        for var in self.output_variables:
            print(var)

            self.output[var + self.postfix] = input_arr.sel({self.split_dim: var})

        return self

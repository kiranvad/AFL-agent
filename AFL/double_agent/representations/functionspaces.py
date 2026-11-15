"""Phase mapping in function spaces using elastic (amplitude-phase) k-means.

Measurements such as SAXS/SANS or UV-Vis spectra are treated as one-dimensional
functions. Elastic k-means clusters them with the amplitude distance after
optimal alignment (warping) of the function domain, so curves that differ only by
a shift or stretch of their features end up in the same phase. Cluster
assignments are smoothed over a k-nearest-neighbor graph of the compositions
(diffusion maps), which favors spatially contiguous phase regions.

The clustering itself is provided by the optional ``autophasemap`` package:

    Vaddi, K., Li, K., & Pozzo, L. D. "Metric geometry tools for automatic
    structure phase map generation." Digital Discovery 2, 1471-1483 (2023).
    https://pubs.rsc.org/en/content/articlelanding/2023/dd/d3dd00105a

Install with ``pip install "AFL-agent[autophasemap]"`` (needs a C compiler).
"""

from __future__ import annotations

import contextlib
import logging
import warnings
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr
from scipy.integrate import trapezoid
from typing_extensions import Self

from AFL.double_agent.PipelineOp import PipelineOp

try:
    import ray
    from autophasemap import compute_BIC, compute_elastic_kmeans
    from sklearn.cluster import kmeans_plusplus
except ImportError as e:
    # Keep this op optional so AgentDriver can start without autophasemap/ray.
    _AUTOPHASEMAP_IMPORT_ERROR = e
    ray = None
    compute_BIC = None
    compute_elastic_kmeans = None
    kmeans_plusplus = None
else:
    _AUTOPHASEMAP_IMPORT_ERROR = None

# Minimum number of points along the function domain (spline derivatives and
# dynamic-programming alignment need a handful of points).
_MIN_DOMAIN_POINTS = 8
# autophasemap's constrained assignment puts at least two samples in every cluster.
_MIN_CLUSTER_SIZE = 2


# ----------------------------------------------------------------------------------
# Helpers. None of these need autophasemap or ray.
# ----------------------------------------------------------------------------------


@dataclass
class _FunctionalData:
    """Data container in the form autophasemap expects.

    ``compute_elastic_kmeans`` only reads ``t``, ``F``, ``C``, ``N`` and ``n_domain``,
    so this stands in for ``autophasemap.BaseDataSet`` without subclassing it.
    """

    t: np.ndarray  # (n_domain,) domain mapped onto [0, 1]
    F: List[np.ndarray]  # N functions, each (n_domain,)
    C: np.ndarray  # (N, n_components) compositions

    @property
    def N(self) -> int:
        return len(self.F)

    @property
    def n_domain(self) -> int:
        return self.t.shape[0]


def _ensure_autophasemap_available() -> None:
    if _AUTOPHASEMAP_IMPORT_ERROR is not None:
        raise RuntimeError(
            (
                "ElasticKMeansClustering requires the optional dependency 'autophasemap'.\n"
                'Install with: pip install "AFL-agent[autophasemap]" (needs a C compiler)\n'
                "Original import error: {}"
            ).format(str(_AUTOPHASEMAP_IMPORT_ERROR))
        )


def _unit_domain(x: np.ndarray, rtol: float = 1e-3) -> np.ndarray:
    """Map a strictly increasing domain coordinate onto [0, 1].

    Warns when the grid is not uniformly spaced; resample first (e.g. with
    ``SavgolFilter``, which produces a uniform grid in log(q)).
    """
    x = np.asarray(x, dtype=float)
    dx = np.diff(x)
    if np.any(dx <= 0):
        raise ValueError("The function domain coordinate must be strictly increasing")
    if not np.allclose(dx, dx.mean(), rtol=rtol):
        warnings.warn(
            "The function domain is not uniformly spaced; consider resampling it first "
            "(e.g. with SavgolFilter)",
            stacklevel=3,
        )
    return (x - x[0]) / (x[-1] - x[0])


def _normalize_functions(F: np.ndarray, t: np.ndarray, method: Optional[str]) -> np.ndarray:
    """Normalize each function to unit L2 norm (``"l2"``), unit maximum (``"max"``) or not at all (``None``)."""
    if method is None:
        return F
    if method == "l2":
        scale = np.sqrt(trapezoid(F**2, t, axis=1))
    elif method == "max":
        scale = np.abs(F).max(axis=1)
    else:
        raise ValueError(f"Unknown normalize={method!r}; use 'l2', 'max' or None")
    if np.any(scale == 0):
        raise ValueError("Cannot normalize a function that is zero everywhere")
    return F / scale[:, None]


def _minmax_scale(C: np.ndarray) -> np.ndarray:
    """Scale each composition component to [0, 1] so no component dominates the kNN graph."""
    span = np.ptp(C, axis=0)
    return (C - C.min(axis=0)) / np.where(span > 0, span, 1.0)


def _validate_problem(
    n_samples: int,
    n_domain: int,
    candidates: Sequence[int],
    n_neighbors: Optional[int],
    num_freq: Optional[int],
) -> None:
    """Check that the problem can be clustered.

    ``n_neighbors`` and ``num_freq`` only matter for diffusion smoothing; pass ``None``
    to skip their checks when smoothing is off.
    """
    if n_domain < _MIN_DOMAIN_POINTS:
        raise ValueError(
            f"Need at least {_MIN_DOMAIN_POINTS} points along the function domain, got {n_domain}"
        )
    if min(candidates) < 2:
        raise ValueError("n_phases must be at least 2")
    if n_samples < _MIN_CLUSTER_SIZE * max(candidates):
        raise ValueError(
            f"{n_samples} samples cannot support {max(candidates)} phases "
            f"(each phase needs at least {_MIN_CLUSTER_SIZE} samples)"
        )
    if n_neighbors is not None and n_neighbors >= n_samples:
        raise ValueError(
            f"n_neighbors ({n_neighbors}) must be smaller than the number of samples ({n_samples})"
        )
    if num_freq is not None and num_freq >= n_samples:
        warnings.warn(
            f"num_freq ({num_freq}) >= number of samples ({n_samples}): "
            "diffusion smoothing will have no effect",
            stacklevel=3,
        )


@contextlib.contextmanager
def _ray_session(
    num_cpus: Optional[int], address: Optional[str], keep_alive: bool
) -> Iterator[None]:
    """Start ray if it is not running; only shut down an instance started here."""
    started = False
    if not ray.is_initialized():
        ray.init(
            address=address,
            num_cpus=num_cpus,
            include_dashboard=False,
            log_to_driver=False,
            ignore_reinit_error=True,
        )
        started = True
    try:
        yield
    finally:
        if started and not keep_alive:
            ray.shutdown()


@contextlib.contextmanager
def _quiet_logger(name: str = "pygsp", level: int = logging.WARNING) -> Iterator[None]:
    """Temporarily raise a logger's level (pygsp logs every kNN graph construction)."""
    logger = logging.getLogger(name)
    old_level = logger.level
    logger.setLevel(level)
    try:
        yield
    finally:
        logger.setLevel(old_level)


def _inertia(res) -> float:
    """Elastic k-means objective: sum of squared amplitude distances to the assigned templates."""
    d_amplitude = np.asarray(res.d_amplitude)
    labels = np.asarray(res.delta_n)
    return float(np.sum(d_amplitude[np.arange(labels.size), labels] ** 2))


def _fit_n_phases(
    data: _FunctionalData,
    n_phases: int,
    n_init: int,
    random_state: np.random.RandomState,
    fit_kwargs: Dict,
):
    """Run elastic k-means from ``n_init`` k-means++ starts and keep the best run.

    The best run has the lowest k-means objective (:func:`_inertia`). autophasemap's
    ``error`` is not used: it is the template change in the last iteration, a
    convergence measure that says nothing about the quality of the clustering.

    Returns the autophasemap result of the best run and its BIC (``inf`` if BIC
    cannot be computed).
    """
    best, best_inertia = None, np.inf
    for _ in range(n_init):
        _, init_idx = kmeans_plusplus(data.C, n_clusters=n_phases, random_state=random_state)
        res = compute_elastic_kmeans(
            data, init_idx, verbose=0, random_state=random_state, **fit_kwargs
        )
        inertia = _inertia(res)
        if best is None or inertia < best_inertia:
            best, best_inertia = res, inertia
    try:
        bic = float(compute_BIC(data, best.fik_gam, best.qik_gam, best.delta_n))
    except Exception as exc:  # degenerate fPCA / singular covariance
        warnings.warn(f"BIC could not be computed for n_phases={n_phases}: {exc}", stacklevel=3)
        bic = np.inf
    return best, bic


def _canonical_order(labels: np.ndarray, n_phases: int) -> np.ndarray:
    """Ordering of phases by decreasing size, so that phase 0 is the most populated."""
    counts = np.bincount(labels, minlength=n_phases)
    return np.argsort(-counts, kind="stable")


def _apply_order(res, order: np.ndarray) -> Dict:
    """Relabel phases with ``order`` and permute every per-phase array consistently.

    ``order[new] = old``, as returned by :func:`_canonical_order`.
    """
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    return dict(
        labels=inverse[np.asarray(res.delta_n)],
        templates=np.asarray(res.templates)[order],
        d_amplitude=np.asarray(res.d_amplitude)[:, order],
        dist=np.asarray(res.dist)[:, order],
        fik_gam=np.asarray(res.fik_gam)[:, order],
        gam_ik=np.asarray(res.gam_ik)[:, order],
        error=float(res.error),
        inertia=_inertia(res),
    )


def _distance_to_membership(dist: np.ndarray, temperature: float) -> np.ndarray:
    """Soft phase membership ``softmax(-dist / temperature)`` over phases; rows sum to 1."""
    z = -(dist - dist.min(axis=1, keepdims=True)) / temperature
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


# ----------------------------------------------------------------------------------
# PipelineOp
# ----------------------------------------------------------------------------------


class ElasticKMeansClustering(PipelineOp):
    """Automatic phase mapping by elastic k-means clustering of functional data.

    Each measurement (e.g. a scattering curve or spectrum) is treated as a 1D
    function. Elastic k-means learns one template function per phase and assigns
    each sample to the template with the smallest amplitude distance after aligning
    the sample to it (phase/warping differences along the domain are factored out).
    Assignments are smoothed over a k-nearest-neighbor graph of the compositions.
    The number of phases can be fixed or selected from candidates by minimum BIC.

    Uses the optional ``autophasemap`` package (``pip install "AFL-agent[autophasemap]"``);
    see Vaddi, Li & Pozzo, Digital Discovery (2023), d3dd00105a.

    Phases are numbered by decreasing size: phase 0 has the most samples.

    Parameters
    ----------
    functions_variable : str
        Variable with the functions, dims ``(sample_dim, domain_dim)``. The domain
        should be uniformly spaced; ``SavgolFilter`` output (uniform in log q) works well.
    components_variable : str
        Variable with the compositions, dims ``(sample_dim, component_dim)``. Used for
        the k-means++ initialization and the diffusion smoothing graph.
    output_prefix : str, default="elastic_kmeans"
        Prefix of all output variables (see Notes).
    sample_dim : str, default="sample"
        Dimension that indexes samples.
    domain_dim : str, optional
        Dimension of the function domain. Inferred as the non-sample dimension of
        ``functions_variable`` when not given.
    component_dim : str, default="component"
        Dimension that indexes composition components.
    n_phases : int or list of int, default=3
        Number of phases, or candidate numbers of phases; with a list, the candidate
        with the lowest BIC is selected.
    n_init : int, default=3
        Number of k-means++ initializations per number of phases; the run with the
        lowest sum of squared elastic distances to its templates is kept.
    max_iter : int, default=50
        Maximum number of elastic k-means iterations per initialization.
    threshold : float, default=1e-2
        Relative template change below which iterations stop.
    smoothen : bool, default=True
        Smooth the distances over the composition graph before assignment.
    n_neighbors : int, default=3
        Number of nearest neighbors of the composition graph.
    num_freq : int, default=30
        Number of graph Fourier modes kept when smoothing.
    grid_dim : int, default=7
        Search neighborhood of the dynamic-programming alignment.
    lam : float, default=0.0
        Regularization of the alignment (penalizes large warping).
    normalize : {"l2", "max", None}, default="l2"
        Per-function normalization applied before clustering.
    scale_components : bool, default=True
        Min-max scale compositions to [0, 1] before building the kNN graph.
    membership_temperature : float, optional
        When given, also output soft memberships ``softmax(-distance / temperature)``.
    store_alignment : bool, default=False
        Also output aligned functions and warping functions (``N x n_phases x n_domain``).
    seed : int, optional, default=0
        Seed for initialization; ``None`` gives a different result every run.
    num_cpus : int, optional
        CPUs for ray when this op starts it (default: all).
    ray_address : str, optional
        Address of an existing ray cluster to connect to.
    keep_ray_alive : bool, default=True
        Leave a ray instance started by this op running for later calls. A ray
        instance started elsewhere is never shut down.
    name : str, default="ElasticKMeansClustering"
        The name to use when added to a Pipeline.

    Notes
    -----
    Outputs, with ``p`` the ``output_prefix`` and phase dimension ``p_phase``:

    - ``p_labels`` (sample): phase label of each sample; attrs ``n_phases``, ``inertia``
      (sum of squared elastic distances to the templates), ``error`` (last template
      change) and ``bic``
    - ``p_templates`` (p_phase, domain): template function of each phase
    - ``p_amplitude_distance`` (sample, p_phase): elastic amplitude distance to each template
    - ``p_distance`` (sample, p_phase): distance after diffusion smoothing (used for labels)
    - ``p_bic`` (p_n_phases): BIC of each candidate number of phases
    - ``p_membership`` (sample, p_phase): only with ``membership_temperature``
    - ``p_aligned``, ``p_warping`` (sample, p_phase, domain): only with ``store_alignment``

    Elastic k-means aligns every sample to every template in each iteration, so it
    is much slower than Euclidean clustering: about 30 s for 100 samples x 100 domain
    points and 2 phases. Downsample the domain (e.g. ``SavgolFilter(npts=100)``) and
    keep ``max_iter``/``n_init`` small inside active-learning loops.

    Examples
    --------
    >>> with Pipeline("phase_mapping") as p:
    ...     SavgolFilter(input_variable="measurement", output_variable="log_measurement",
    ...                  dim="x", npts=100, apply_log_scale=True)
    ...     ElasticKMeansClustering(functions_variable="log_measurement",
    ...                             components_variable="composition",
    ...                             domain_dim="log_x", n_phases=2)
    """

    def __init__(
        self,
        functions_variable: str,
        components_variable: str,
        output_prefix: str = "elastic_kmeans",
        sample_dim: str = "sample",
        domain_dim: Optional[str] = None,
        component_dim: str = "component",
        n_phases: Union[int, List[int]] = 3,
        n_init: int = 3,
        max_iter: int = 50,
        threshold: float = 1e-2,
        smoothen: bool = True,
        n_neighbors: int = 3,
        num_freq: int = 30,
        grid_dim: int = 7,
        lam: float = 0.0,
        normalize: Optional[str] = "l2",
        scale_components: bool = True,
        membership_temperature: Optional[float] = None,
        store_alignment: bool = False,
        seed: Optional[int] = 0,
        num_cpus: Optional[int] = None,
        ray_address: Optional[str] = None,
        keep_ray_alive: bool = True,
        name: str = "ElasticKMeansClustering",
    ) -> None:
        super().__init__(
            name=name,
            input_variable=[functions_variable, components_variable],
            output_prefix=output_prefix,
        )
        self.functions_variable = functions_variable
        self.components_variable = components_variable
        self.sample_dim = sample_dim
        self.domain_dim = domain_dim
        self.component_dim = component_dim
        self.n_phases = n_phases
        self.n_init = n_init
        self.max_iter = max_iter
        self.threshold = threshold
        self.smoothen = smoothen
        self.n_neighbors = n_neighbors
        self.num_freq = num_freq
        self.grid_dim = grid_dim
        self.lam = lam
        self.normalize = normalize
        self.scale_components = scale_components
        self.membership_temperature = membership_temperature
        self.store_alignment = store_alignment
        self.seed = seed
        self.num_cpus = num_cpus
        self.ray_address = ray_address
        self.keep_ray_alive = keep_ray_alive

        # result of the last calculate() with phases in output order
        self.result_: Optional[Dict] = None
        self._banned_from_attrs.extend(["result_"])

    def _candidates(self) -> List[int]:
        if np.isscalar(self.n_phases):
            return [int(self.n_phases)]
        return sorted({int(k) for k in self.n_phases})

    def _extract(self, dataset: xr.Dataset) -> Tuple[xr.DataArray, xr.DataArray, str]:
        functions = dataset[self.functions_variable]
        domain_dim = self.domain_dim
        if domain_dim is None:
            others = [d for d in functions.dims if d != self.sample_dim]
            if len(others) != 1:
                raise ValueError(
                    f"Cannot infer domain_dim from dims {functions.dims}; pass domain_dim"
                )
            domain_dim = others[0]
        functions = functions.transpose(self.sample_dim, domain_dim)
        components = dataset[self.components_variable].transpose(
            self.sample_dim, self.component_dim
        )
        if functions.isnull().any() or components.isnull().any():
            raise ValueError("ElasticKMeansClustering inputs contain NaN values")
        return functions, components, domain_dim

    def _build_data(
        self, functions: xr.DataArray, components: xr.DataArray, domain_dim: str
    ) -> _FunctionalData:
        t = _unit_domain(functions[domain_dim].values)
        F = _normalize_functions(functions.values.astype(float), t, self.normalize)
        C = components.values.astype(float)
        if self.scale_components:
            C = _minmax_scale(C)
        return _FunctionalData(t=t, F=list(F), C=C)

    def _fit_kwargs(self) -> Dict:
        return dict(
            max_iter=self.max_iter,
            threshold=self.threshold,
            smoothen=self.smoothen,
            k=self.n_neighbors,
            num_freq=self.num_freq,
            grid_dim=self.grid_dim,
            lam=self.lam,
        )

    def calculate(self, dataset: xr.Dataset) -> Self:
        """Cluster the functions in ``dataset`` into phases."""
        _ensure_autophasemap_available()
        functions, components, domain_dim = self._extract(dataset)
        data = self._build_data(functions, components, domain_dim)
        candidates = self._candidates()
        _validate_problem(
            data.N,
            data.n_domain,
            candidates,
            self.n_neighbors if self.smoothen else None,
            self.num_freq if self.smoothen else None,
        )

        random_state = np.random.RandomState(self.seed)
        fits, bics = {}, {}
        with _ray_session(self.num_cpus, self.ray_address, self.keep_ray_alive), _quiet_logger():
            for n_phases in candidates:
                fits[n_phases], bics[n_phases] = _fit_n_phases(
                    data, n_phases, self.n_init, random_state, self._fit_kwargs()
                )

        best = min(bics, key=bics.get) if len(candidates) > 1 else candidates[0]
        order = _canonical_order(np.asarray(fits[best].delta_n), best)
        self.result_ = _apply_order(fits[best], order)
        self._write_outputs(self.result_, functions, domain_dim, best, bics)
        return self

    def _write_outputs(
        self,
        res: Dict,
        functions: xr.DataArray,
        domain_dim: str,
        n_phases: int,
        bics: Dict[int, float],
    ) -> None:
        sample_dim = self.sample_dim
        phase_dim = f"{self.output_prefix}_phase"
        n_phases_dim = f"{self.output_prefix}_n_phases"
        sample_coords = (
            {sample_dim: functions[sample_dim].values} if sample_dim in functions.coords else {}
        )
        phase_coords = {phase_dim: np.arange(n_phases)}
        per_phase = dict(dims=[sample_dim, phase_dim], coords={**sample_coords, **phase_coords})

        self.output[self._prefix_output("labels")] = xr.DataArray(
            res["labels"],
            dims=[sample_dim],
            coords=sample_coords,
            attrs={
                "n_phases": n_phases,
                "inertia": res["inertia"],
                "error": res["error"],
                "bic": bics[n_phases],
            },
        )
        self.output[self._prefix_output("templates")] = xr.DataArray(
            res["templates"],
            dims=[phase_dim, domain_dim],
            coords={**phase_coords, domain_dim: functions[domain_dim].values},
        )
        self.output[self._prefix_output("amplitude_distance")] = xr.DataArray(
            res["d_amplitude"], **per_phase
        )
        self.output[self._prefix_output("distance")] = xr.DataArray(res["dist"], **per_phase)
        self.output[self._prefix_output("bic")] = xr.DataArray(
            list(bics.values()), dims=[n_phases_dim], coords={n_phases_dim: list(bics.keys())}
        )
        if self.membership_temperature is not None:
            self.output[self._prefix_output("membership")] = xr.DataArray(
                _distance_to_membership(res["dist"], self.membership_temperature), **per_phase
            )
        if self.store_alignment:
            per_point = dict(
                dims=[sample_dim, phase_dim, domain_dim],
                coords={**sample_coords, **phase_coords, domain_dim: functions[domain_dim].values},
            )
            self.output[self._prefix_output("aligned")] = xr.DataArray(res["fik_gam"], **per_point)
            self.output[self._prefix_output("warping")] = xr.DataArray(res["gam_ik"], **per_point)

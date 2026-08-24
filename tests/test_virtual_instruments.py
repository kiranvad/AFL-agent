import numpy as np
import pytest

from AFL.double_agent.VirtualInstruments import GaussianVirtualInstrument


def test_gaussian_virtual_instrument_measures_batches(tmp_path):
    instrument = GaussianVirtualInstrument(afl_home=tmp_path)
    result = instrument.measure([[0.25, 0.05], [0.75, 0.10]])

    assert result["parameters"].dims == ("sample", "component")
    assert result["spectrum"].dims == ("sample", "wavelength")
    assert result["spectrum"].shape == (2, 201)
    np.testing.assert_array_equal(result["component"], ["mu", "sigma"])
    first_peak = result["spectrum"].isel(sample=0).argmax(dim="wavelength")
    second_peak = result["spectrum"].isel(sample=1).argmax(dim="wavelength")
    assert result["wavelength"].isel(wavelength=first_peak).item() == pytest.approx(0.25)
    assert result["wavelength"].isel(wavelength=second_peak).item() == pytest.approx(0.75)


@pytest.mark.parametrize(
    "parameters, message",
    [
        ([0.5], "shape"),
        ([[np.nan, 0.1]], "finite"),
        ([[0.5, 0.0]], "greater than zero"),
    ],
)
def test_gaussian_virtual_instrument_rejects_invalid_parameters(tmp_path, parameters, message):
    instrument = GaussianVirtualInstrument(afl_home=tmp_path)
    with pytest.raises(ValueError, match=message):
        instrument.measure(parameters)

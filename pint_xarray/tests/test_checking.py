import pytest
import pint
import xarray as xr

from pint import UnitRegistry

from ..checking import expects

ureg = UnitRegistry()


class TestExpects:
    def test_single_arg(self):

        @expects('degC')
        def above_freezing(temp : pint.Quantity):
            return temp.magnitude > 0

        f_q = pint.Quantity(20, units='degF')
        assert above_freezing(f_q) == False

        c_q = pint.Quantity(-2, units='degC')
        assert above_freezing(c_q) == False

        @expects('degC')
        def above_freezing(temp : xr.DataArray):
            return temp.pint.magnitude > 0

        f_da = xr.DataArray(20).pint.quantify(units='degF')
        assert above_freezing(f_da) == False

        c_da = xr.DataArray(-2).pint.quantify(units='degC')
        assert above_freezing(c_da) == False

    def test_single_kwarg(self):

        @expects('meters', c='meters / second')
        def freq(wavelength, c=None):
            if c is None:
                c = ureg.speed_of_light

            return c / wavelength

    def test_single_return_value(self):

        @expects('kg', 'm / s^2', return_units='newtons')
        def second_law(m, a):
            return m * a

        m_q = pint.Quantity(0.1, units='tons')
        a_q = pint.Quantity(10, units='feet / second^2')
        assert second_law(m_q, a_q).pint.units == pint.Unit('newtons')

        m_da
        a_da

    @pytest.mark.xfail
    def test_multiple_return_values(self):
        raise NotImplementedError

    @pytest.mark.xfail
    def test_mixed_args_kwargs_return_values(self):
        raise NotImplementedError

    @pytest.mark.xfail
    def test_invalid_input_types(self):
        raise NotImplementedError

    @pytest.mark.xfail
    def test_invalid_return_types(self):
        raise NotImplementedError

    @pytest.mark.xfail
    def test_unquantified_arrays(self):
        raise NotImplementedError

    @pytest.mark.xfail
    def test_wrong_number_of_args(self):
        raise NotImplementedError

    @pytest.mark.xfail
    def test_nonexistent_kwarg(self):
        raise NotImplementedError

    @pytest.mark.xfail
    def test_unexpected_return_value(self):
        raise NotImplementedError

    @pytest.mark.xfail
    def test_expected_return_value(self):
        raise NotImplementedError

    @pytest.mark.xfail
    def test_wrong_number_of_return_values(self):
        raise NotImplementedError

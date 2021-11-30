import pint
import pytest
import xarray as xr
from pint import UnitRegistry

from ..checking import expects

ureg = UnitRegistry()


class TestExpects:
    def test_single_arg(self):
        @expects("degC")
        def above_freezing(temp):
            return temp > 0

        f_q = pint.Quantity(20, units="degF")
        assert not above_freezing(f_q)

        c_q = pint.Quantity(-2, units="degC")
        assert not above_freezing(c_q)

        f_da = xr.DataArray(20).pint.quantify(units="degF")
        assert not above_freezing(f_da)

        c_da = xr.DataArray(-2).pint.quantify(units="degC")
        assert not above_freezing(c_da)

    def test_single_kwarg(self):
        @expects("meters", c="meters / second", return_units="Hz")
        def freq(wavelength, c=None):
            if c is None:
                c = (1 * ureg.speed_of_light).to("meters / second").magnitude

            return c / wavelength

        w_q = pint.Quantity(0.02, units="inches")
        c_q = pint.Quantity(1e6, units="feet / second")
        f_q = freq(w_q, c=c_q)
        assert f_q.units == pint.Unit("hertz")
        f_q = freq(w_q)
        assert f_q.units == pint.Unit("hertz")

        w_da = xr.DataArray(0.02).pint.quantify(units="inches")
        c_da = xr.DataArray(1e6).pint.quantify(units="feet / second")
        f_da = freq(w_da, c=c_da)
        assert f_da.pint.units == pint.Unit("hertz")
        f_da = freq(w_da)
        assert f_da.pint.units == pint.Unit("hertz")

    def test_single_return_value(self):
        @expects("kg", "m / s^2", return_units="newtons")
        def second_law(m, a):
            return m * a

        m_q = pint.Quantity(0.1, units="tons")
        a_q = pint.Quantity(10, units="feet / second^2")
        expected_q = (m_q * a_q).to("newtons")
        result_q = second_law(m_q, a_q)
        assert result_q == expected_q

        m_da = xr.DataArray(0.1).pint.quantify(units="tons")
        a_da = xr.DataArray(10).pint.quantify(units="feet / second^2")
        expected_da = (m_da * a_da).pint.to("newtons")
        result_da = second_law(m_da, a_da)
        assert result_da == expected_da

    def test_multiple_return_values(self):
        @expects("kg", "m / s", return_units=["J", "newton seconds"])
        def energy_and_momentum(m, v):
            ke = 0.5 * m * v ** 2
            p = m * v
            return ke, p

        m = pint.Quantity(0.1, units="tons")
        v = pint.Quantity(10, units="feet / second")
        expected_ke = (0.5 * m * v ** 2).to("J")
        expected_p = (m * v).to("newton seconds")
        result_ke, result_p = energy_and_momentum(m, v)
        assert result_ke.units == expected_ke.units
        assert result_p.units == expected_p.units

        m = xr.DataArray(0.1).pint.quantify(units="tons")
        v = xr.DataArray(10).pint.quantify(units="feet / second")
        expected_ke = (0.5 * m * v ** 2).pint.to("J")
        expected_p = (m * v).pint.to("newton seconds")
        result_ke, result_p = energy_and_momentum(m, v)
        assert result_ke.pint.units == expected_ke.pint.units
        assert result_p.pint.units == expected_p.pint.units

    def test_dont_check_arg_units(self):
        @expects("seconds", None, return_units=None)
        def finite_difference(a, type):
            return ...

        t = pint.Quantity(0.1, units="seconds")
        finite_difference(t, "centered")

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

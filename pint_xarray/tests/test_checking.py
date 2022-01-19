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

    @pytest.mark.parametrize(
        "arg_units, return_units",
        [("nonsense", "Hertz"), ("seconds", 6), ("seconds", [6])],
    )
    def test_invalid_unit_types(self, arg_units, return_units):
        @expects(arg_units, return_units=return_units)
        def freq(period):
            return 1 / period

        q = pint.Quantity(1.0, units="seconds")

        with pytest.raises((TypeError, pint.errors.UndefinedUnitError)):
            freq(q)

    @pytest.mark.xfail
    def test_unquantified_arrays(self):
        raise NotImplementedError

    def test_wrong_number_of_args(self):
        with pytest.raises(
            TypeError,
            match="expects 1 arguments, but a function expecting 2 arguments was wrapped",
        ):

            @expects("kg", return_units="newtons")
            def second_law(m, a):
                return m * a

    def test_wrong_number_of_return_values(self):
        @expects("kg", "m / s^2", return_units=["newtons", "joules"])
        def second_law(m, a):
            return m * a

        m_q = pint.Quantity(0.1, units="tons")
        a_q = pint.Quantity(10, units="feet / second^2")

        with pytest.raises(TypeError, match="2 return values were expected"):
            second_law(m_q, a_q)

    def test_expected_return_value(self):
        @expects("seconds", return_units="Hz")
        def freq(period):
            return None

        p = pint.Quantity(2, units="seconds")

        with pytest.raises(TypeError, match="function returned None"):
            freq(p)

    def test_input_unit_dict(self):
        @expects({"m": "kg", "a": "m / s^2"}, return_units="newtons")
        def second_law(ds):
            return ds["m"] * ds["a"]

        m_da = xr.DataArray(0.1).pint.quantify(units="tons")
        a_da = xr.DataArray(10).pint.quantify(units="feet / second^2")
        ds = xr.Dataset({"m": m_da, "a": a_da})

        expected_da = (m_da * a_da).pint.to("newtons")
        result_da = second_law(ds)
        assert result_da == expected_da

    def test_return_dataset(self):
        @expects({"m": "kg", "a": "m / s^2"}, return_units=[{"f": "newtons"}])
        def second_law(ds):
            f_da = ds["m"] * ds["a"]
            return xr.Dataset({"f": f_da})

        m_da = xr.DataArray(0.1).pint.quantify(units="tons")
        a_da = xr.DataArray(10).pint.quantify(units="feet / second^2")
        ds = xr.Dataset({"m": m_da, "a": a_da})

        expected_da = m_da * a_da
        expected_ds = xr.Dataset({"f": expected_da}).pint.to({"f": "newtons"})
        result_ds = second_law(ds)
        assert result_ds == expected_ds

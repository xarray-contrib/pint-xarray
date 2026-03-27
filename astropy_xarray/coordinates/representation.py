import astropy.units as u
from astropy.coordinates import (
    CartesianDifferential,
    CartesianRepresentation,
    CylindricalRepresentation,
    EarthLocation,
    SphericalRepresentation,
)

from astropy_xarray.coordinates.core import dump_quantity


# Representation
def dump_spherical_representation(rep: SphericalRepresentation):
    return {
        "lon": dump_quantity(rep.lon),
        "lat": dump_quantity(rep.lat),
        "distance": dump_quantity(rep.distance),
    }


def dump_cartesian_representation(rep: CartesianRepresentation):
    return {
        "x": dump_quantity(rep.x),
        "y": dump_quantity(rep.y),
        "z": dump_quantity(rep.z),
    }


def load_cartesian_representation(kwargs: dict):
    return CartesianRepresentation(
        x=u.Quantity(**kwargs["x"]),
        y=u.Quantity(**kwargs["y"]),
        z=u.Quantity(**kwargs["z"]),
    )


def dump_cylindrical_representation(rep: CylindricalRepresentation):
    return {
        "rho": dump_quantity(rep.rho),
        "phi": dump_quantity(rep.phi),
        "z": dump_quantity(rep.z),
    }


# Earth Location
def dump_earth_location(el: EarthLocation):
    return {
        "x": dump_quantity(el.x),
        "y": dump_quantity(el.y),
        "z": dump_quantity(el.z),
    }


def load_optional_earthlocation(kwargs: dict | None) -> EarthLocation | None:
    return (
        EarthLocation(
            x=u.Quantity(**kwargs["x"]),
            y=u.Quantity(**kwargs["y"]),
            z=u.Quantity(**kwargs["z"]),
        )
        if kwargs is not None
        else None
    )


# Cartesian Differential
def dump_cartesian_differential(cd: CartesianDifferential):
    return {
        "d_x": dump_quantity(cd.d_x),
        "d_y": dump_quantity(cd.d_y),
        "d_z": dump_quantity(cd.d_z),
    }


def load_cartesian_differential(kwargs: dict):
    return CartesianDifferential(
        d_x=u.Quantity(**kwargs["d_x"]),
        d_y=u.Quantity(**kwargs["d_y"]),
        d_z=u.Quantity(**kwargs["d_z"]),
    )

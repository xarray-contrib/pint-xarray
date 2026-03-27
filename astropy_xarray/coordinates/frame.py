from types import NoneType

import astropy.units as u
import numpy as np
from astropy.coordinates import (
    BaseCoordinateFrame,
    BaseRepresentation,
    CartesianDifferential,
    CartesianRepresentation,
    CylindricalRepresentation,
    EarthLocation,
    SphericalRepresentation,
    frame_transform_graph,
    representation,
)
from astropy.coordinates.builtin_frames import (
    CIRS,
    FK4,
    FK5,
    GCRS,
    HCRS,
    ICRS,
    ITRS,
    AltAz,
    CustomBarycentricEcliptic,
    Galactic,
    GalacticLSR,
    Galactocentric,
    GeocentricMeanEcliptic,
    GeocentricTrueEcliptic,
    HADec,
    HeliocentricEclipticIAU76,
    HeliocentricMeanEcliptic,
    HeliocentricTrueEcliptic,
    PrecessedGeocentric,
    Supergalactic,
)
from astropy.time import Time

from astropy_xarray.coordinates.core import (
    dump_quantity,
    dump_time,
    load_optional_object,
)
from astropy_xarray.coordinates.representation import (
    dump_cartesian_differential,
    dump_cartesian_representation,
    dump_cylindrical_representation,
    dump_earth_location,
    dump_spherical_representation,
    load_cartesian_differential,
    load_cartesian_representation,
    load_optional_earthlocation,
)


def dump_frame(frame: BaseCoordinateFrame, with_data: bool = False) -> dict:
    ser = {
        "name": frame.name,
        "representation_type": frame.representation_type.get_name(),
        "differential_type": frame.differential_type.get_name(),
    }
    if frame.has_data:
        ser["data"] = {"representation_type": frame.data.get_name()}
        if "s" in frame.data.differentials:
            ser["data"]["differential_type"] = frame.data.differentials["s"].get_name()
        if with_data:
            for component in frame.data.components:
                quantity: u.Quantity = getattr(frame.data, component)
                ser["data"][component] = {
                    "value": float(quantity.value),
                    "unit": str(quantity.unit),
                }
            if "s" in frame.data.differentials:
                for component in frame.data.differentials["s"].components:
                    quantity: u.Quantity = getattr(
                        frame.data.differentials["s"], component
                    )
                    ser["data"][component] = {
                        "value": float(quantity.value),
                        "unit": str(quantity.unit),
                    }

    for attribute_name in frame.frame_attributes.keys():
        attr = ser[attribute_name] = getattr(frame, attribute_name)
        if not isinstance(attr, NoneType):
            match attr:
                case BaseCoordinateFrame():
                    ser_attr = dump_frame(attr, with_data=True)
                case Time():
                    ser_attr = dump_time(attr)
                case EarthLocation():
                    ser_attr = dump_earth_location(attr)
                case SphericalRepresentation():
                    ser_attr = dump_spherical_representation(attr)
                case CartesianRepresentation():
                    ser_attr = dump_cartesian_representation(attr)
                case CylindricalRepresentation():
                    ser_attr = dump_cylindrical_representation(attr)
                case CartesianDifferential():
                    ser_attr = dump_cartesian_differential(attr)
                case u.Quantity():
                    ser_attr = dump_quantity(attr)
                case _:
                    raise NotImplementedError("unsupported frame member", attr)
            ser[attribute_name] = ser_attr
    return ser


def load_frame(frame_dict: dict, with_data: bool = False) -> BaseCoordinateFrame:
    representation_type = frame_dict.get("data").get("representation_type")
    differential_type = frame_dict.get("data").get("differential_type")
    kwargs = dict(
        representation_type=representation_type, differential_type=differential_type
    )
    match frame_dict["name"]:
        case "icrs":
            frame = ICRS(**kwargs)
        case "fk5":
            frame = FK5(
                equinox=load_optional_object(Time, frame_dict["equinox"]), **kwargs
            )
        case "fk4":
            frame = FK4(
                equinox=load_optional_object(Time, frame_dict["equinox"]),
                obstime=load_optional_object(Time, frame_dict["obstime"]),
                **kwargs,
            )
        case "galactic":
            frame = Galactic(**kwargs)
        case "galacticlsr":
            frame = GalacticLSR(**kwargs)
        case "galactocentric":
            frame = Galactocentric(
                galcen_distance=u.Quantity(**frame_dict["galcen_distance"]),
                galcen_coord=load_frame(frame_dict["galcen_coord"], with_data=True),
                galcen_v_sun=load_cartesian_differential(frame_dict["galcen_v_sun"]),
                z_sun=u.Quantity(**frame_dict["z_sun"]),
                roll=u.Quantity(**frame_dict["roll"]),
                **kwargs,
            )
        case "gcrs":
            frame = GCRS(
                obstime=load_optional_object(Time, frame_dict["obstime"]),
                obsgeoloc=load_cartesian_representation(frame_dict["obsgeoloc"]),
                obsgeovel=load_cartesian_representation(frame_dict["obsgeovel"]),
                **kwargs,
            )
        case "cirs":
            frame = CIRS(
                obstime=load_optional_object(Time, frame_dict["obstime"]),
                location=load_optional_earthlocation(frame_dict["location"]),
                **kwargs,
            )
        case "itrs":
            frame = ITRS(
                obstime=load_optional_object(Time, frame_dict["obstime"]),
                location=load_optional_earthlocation(frame_dict["location"]),
                **kwargs,
            )
        case "hcrs":
            frame = HCRS(
                obstime=load_optional_object(Time, frame_dict["obstime"]), **kwargs
            )
        case "itrs":
            frame = ITRS(
                location=load_optional_earthlocation(frame_dict["location"]),
                **kwargs,
            )
        case "altaz":
            frame = AltAz(
                obstime=load_optional_object(Time, frame_dict["obstime"]),
                location=load_optional_earthlocation(frame_dict["location"]),
                pressure=load_optional_object(u.Quantity, frame_dict["pressure"]),
                temperature=load_optional_object(u.Quantity, frame_dict["temperature"]),
                relative_humidity=load_optional_object(
                    u.Quantity, frame_dict["relative_humidity"]
                ),
                obswl=load_optional_object(u.Quantity, frame_dict["obswl"]),
                **kwargs,
            )
        case "hadec":
            frame = HADec(
                obstime=load_optional_object(Time, frame_dict["obstime"]),
                location=load_optional_earthlocation(frame_dict["location"]),
                pressure=load_optional_object(u.Quantity, frame_dict["pressure"]),
                temperature=load_optional_object(u.Quantity, frame_dict["temperature"]),
                relative_humidity=load_optional_object(
                    u.Quantity, frame_dict["relative_humidity"]
                ),
                obswl=load_optional_object(u.Quantity, frame_dict["obswl"]),
                **kwargs,
            )
        case "supergalactic":
            frame = Supergalactic(**kwargs)
        case "precessedgeocentric":
            frame = PrecessedGeocentric(
                equinox=load_optional_object(Time, frame_dict["equinox"]),
                obstime=load_optional_object(Time, frame_dict["obstime"]),
                obsgeoloc=load_cartesian_representation(frame_dict["obsgeoloc"]),
                obsgeovel=load_cartesian_representation(frame_dict["obsgeovel"]),
                **kwargs,
            )
        case "geocentricmeanecliptic":
            frame = GeocentricMeanEcliptic(
                equinox=load_optional_object(Time, frame_dict["equinox"]),
                obstime=load_optional_object(Time, frame_dict["obstime"]),
                **kwargs,
            )
        case "geocentrictrueecliptic":
            frame = GeocentricTrueEcliptic(
                equinox=load_optional_object(Time, frame_dict["equinox"]),
                obstime=load_optional_object(Time, frame_dict["obstime"]),
                **kwargs,
            )
        case "heliocentricmeanecliptic":
            frame = HeliocentricMeanEcliptic(
                equinox=load_optional_object(Time, frame_dict["equinox"]),
                obstime=load_optional_object(Time, frame_dict["obstime"]),
                **kwargs,
            )
        case "heliocentrictrueecliptic":
            frame = HeliocentricTrueEcliptic(
                equinox=load_optional_object(Time, frame_dict["equinox"]),
                obstime=load_optional_object(Time, frame_dict["obstime"]),
                **kwargs,
            )
        case "heliocentriceclipticiau76":
            frame = HeliocentricEclipticIAU76(
                obstime=load_optional_object(Time, frame_dict["obstime"]), **kwargs
            )
        case "custombarycentricecliptic":
            frame = CustomBarycentricEcliptic(
                obliquity=load_optional_object(u.Quantity, frame_dict["obliquity"]),
                **kwargs,
            )
        case _:
            raise NotImplementedError(frame_dict["name"])
    if with_data:
        frame = frame.realize_frame(
            representation_type=frame_dict.get("representation_type"),
            differential_type=frame_dict.get("differential_type"),
            data=load_representation(
                representation_type,
                differential_type,
                None,
                {
                    k: u.Quantity(**v)
                    for k, v in frame_dict.get("data").items()
                    if isinstance(v, dict)
                },
            ),
        )
    return frame


def load_representation(
    representation_type: str,
    differential_type: str | None,
    frame_name: str | None,
    data: dict[str, np.ndarray],
) -> BaseRepresentation:
    RepresentationClass = representation.REPRESENTATION_CLASSES.get(representation_type)
    DifferentialClass = representation.DIFFERENTIAL_CLASSES.get(differential_type)

    if frame_name is None:
        # using data component names
        differentials = (
            DifferentialClass(
                **{key: data[key] for key in DifferentialClass.attr_classes.keys()},
                copy=False,
            )
            if DifferentialClass is not None
            else None
        )

        return RepresentationClass(
            **{key: data[key] for key in RepresentationClass.attr_classes.keys()},
            differentials=differentials,
            copy=False,
        )
    else:
        # using frame component names
        frame_type: BaseCoordinateFrame = frame_transform_graph.lookup_name(frame_name)
        diff_to_data = (
            dict(
                zip(
                    tuple(DifferentialClass.attr_classes),
                    frame_type._get_representation_info()[DifferentialClass]["names"],
                )
            )
            if DifferentialClass is not None
            else {}
        )
        rep_to_data = (
            dict(
                zip(
                    tuple(RepresentationClass.attr_classes),
                    frame_type._get_representation_info()[RepresentationClass]["names"],
                )
            )
            if RepresentationClass is not None
            else {}
        )

        differentials = (
            DifferentialClass(
                **{
                    key: data[diff_to_data[key]]
                    for key in DifferentialClass.attr_classes.keys()
                },
                copy=False,
            )
            if DifferentialClass is not None
            else None
        )

        return RepresentationClass(
            **{
                key: data[rep_to_data[key]]
                for key in RepresentationClass.attr_classes.keys()
            },
            differentials=differentials,
            copy=False,
        )

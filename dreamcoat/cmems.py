import os
from datetime import date
import numpy as np
import xarray as xr


def get_surphys_filename(
    date_min, date_max, longitude_min, longitude_max, latitude_min, latitude_max
):
    """Generate filename used to save downloaded CMEMS files from the
    GLOBAL_ANALYSIS_FORECAST_PHY_001_024 dataset.

    Parameters
    ----------
    date_min : str
        First date of data to download in '%Y-%m-%d' format.  Uses today if None.
    date_max : str
        Last date of data to download in '%Y-%m-%d' format.  Uses today if None.
    latitude_min : int
        Minimum latitude in decimal degrees N.
    latitude_max : int
        Maximum latitude in decimal degrees N.
    longitude_min : int
        Minimum longitude in decimal degrees E.
    longitude_max : int
        Maximum longitude in decimal degrees E.

    Returns
    -------
    str
        The filename for the specified date and location ranges.
    """
    if not date_min:
        date_min = date.today().strftime("%Y-%m-%d")
    if not date_max:
        date_max = date.today().strftime("%Y-%m-%d")
    return "global-analysis-forecast-phy-001-024_{}_{}_{}_{}_{}_{}.nc".format(
        date_min, date_max, longitude_min, longitude_max, latitude_min, latitude_max
    )


def get_surbio_filename(
    date_min, date_max, longitude_min, longitude_max, latitude_min, latitude_max
):
    """Generate filename used to save downloaded CMEMS files from the
    GLOBAL_ANALYSIS_FORECAST_BIO_001_028 dataset.

    Parameters
    ----------
    date_min : str
        First date of data to download in '%Y-%m-%d' format.  Uses today if None.
    date_max : str
        Last date of data to download in '%Y-%m-%d' format.  Uses today if None.
    latitude_min : int
        Minimum latitude in decimal degrees N.
    latitude_max : int
        Maximum latitude in decimal degrees N.
    longitude_min : int
        Minimum longitude in decimal degrees E.
    longitude_max : int
        Maximum longitude in decimal degrees E.

    Returns
    -------
    str
        The filename for the specified date and location ranges.
    """
    if not date_min:
        date_min = date.today().strftime("%Y-%m-%d")
    if not date_max:
        date_max = date.today().strftime("%Y-%m-%d")
    return "global-analysis-forecast-bio-001-028_{}_{}_{}_{}_{}_{}.nc".format(
        date_min, date_max, longitude_min, longitude_max, latitude_min, latitude_max
    )


def download_surphys(
    filename=None,
    filepath="",
    date_min=None,
    date_max=None,
    latitude_min=-90,
    latitude_max=90,
    longitude_min=-180,
    longitude_max=180,
    username=None,
    password=None,
):
    """Download a missing CMEMS data file from the GLOBAL_ANALYSIS_FORECAST_PHY_001_024
    dataset including surface fields of salinity (so), potential temperature (thetao),
    mixed layer depth (mlotst), sea surface height (zos), and eastwards (uo) and
    northwards (vo) current velocities.

    Parameters
    ----------
    filename : str, optional
        File name to save to, by default None, in which case it is generated by
        get_surphys_filename().
    filepath : str, optional
        File path for the saved file, by default "".
    date_min : str, optional
        First date of data to download in '%Y-%m-%d' format, by default None, in which
        case today's date is used.
    date_max : str, optional
        Last date of data to download in '%Y-%m-%d' format, by default None, in which
        case today's date is used.
    latitude_min : int, optional
        Minimum latitude in decimal degrees N, by default -90.
    latitude_max : int, optional
        Maximum latitude in decimal degrees N, by default 90.
    longitude_min : int, optional
        Minimum longitude in decimal degrees E, by default -180.
    longitude_max : int, optional
        Maximum longitude in decimal degrees E, by default 180.
    username : str, optional
        Your CMEMS username, by default None, in which case you will be prompted for it.
    password : str, optional
        Your CMEMS password, by default None, in which case you will be prompted for it.
    """
    # Deal with None inputs
    if not username:
        username = input("Please enter your CMEMS username: ")
    if not password:
        password = input("Please enter your CMEMS password: ")
    if not date_min:
        date_min = date.today().strftime("%Y-%m-%d")
    if not date_max:
        date_max = date.today().strftime("%Y-%m-%d")
    if not filename:
        filename = get_surphys_filename(
            date_min, date_max, longitude_min, longitude_max, latitude_min, latitude_max
        )
    # Download the file
    os.system(
        "motuclient --motu https://nrt.cmems-du.eu/motu-web/Motu "
        + "--service-id GLOBAL_ANALYSIS_FORECAST_PHY_001_024-TDS "
        + "--product-id global-analysis-forecast-phy-001-024 "
        + "--longitude-min {} ".format(longitude_min)
        + "--longitude-max {} ".format(longitude_max)
        + "--latitude-min {} ".format(latitude_min)
        + "--latitude-max {} ".format(latitude_max)
        + '--date-min "{} 12:00:00" '.format(date_min)
        + '--date-max "{} 12:00:00" '.format(date_max)
        + "--depth-min 0.494 --depth-max 0.4941 "
        + "--variable mlotst --variable so --variable thetao "
        + "--variable uo --variable vo --variable zos "
        + "--out-dir {} ".format(filepath)
        + "--out-name {} ".format(filename)
        + "--user {} --pwd {}".format(username, password)
    )


def download_surbio(
    filename=None,
    filepath="",
    date_min=None,
    date_max=None,
    latitude_min=-90,
    latitude_max=90,
    longitude_min=-180,
    longitude_max=180,
    username=None,
    password=None,
):
    """Download a missing CMEMS data file from the GLOBAL_ANALYSIS_FORECAST_BIO_001_028
    # dataset including surface fields of salinity (so), potential temperature (thetao),
    # mixed layer depth (mlotst), sea surface height (zos), and eastwards (uo) and
    # northwards (vo) current velocities.

    Parameters
    ----------
    filename : str, optional
        File name to save to, by default None, in which case it is generated by
        get_surbio_filename().
    filepath : str, optional
        File path for the saved file, by default "".
    date_min : str, optional
        First date of data to download in '%Y-%m-%d' format, by default None, in which
        case today's date is used.
    date_max : str, optional
        Last date of data to download in '%Y-%m-%d' format, by default None, in which
        case today's date is used.
    latitude_min : int, optional
        Minimum latitude in decimal degrees N, by default -90.
    latitude_max : int, optional
        Maximum latitude in decimal degrees N, by default 90.
    longitude_min : int, optional
        Minimum longitude in decimal degrees E, by default -180.
    longitude_max : int, optional
        Maximum longitude in decimal degrees E, by default 180.
    username : str, optional
        Your CMEMS username, by default None, in which case you will be prompted for it.
    password : str, optional
        Your CMEMS password, by default None, in which case you will be prompted for it.
    """
    # Deal with None inputs
    if not username:
        username = input("Please enter your CMEMS username: ")
    if not password:
        password = input("Please enter your CMEMS password: ")
    if not date_min:
        date_min = date.today().strftime("%Y-%m-%d")
    if not date_max:
        date_max = date.today().strftime("%Y-%m-%d")
    if not filename:
        filename = get_surbio_filename(
            date_min, date_max, longitude_min, longitude_max, latitude_min, latitude_max
        )
    # Download the file
    os.system(
        "motuclient --motu https://nrt.cmems-du.eu/motu-web/Motu "
        + "--service-id GLOBAL_ANALYSIS_FORECAST_BIO_001_028-TDS "
        + "--product-id global-analysis-forecast-bio-001-028-daily "
        + "--longitude-min {} ".format(longitude_min)
        + "--longitude-max {} ".format(longitude_max)
        + "--latitude-min {} ".format(latitude_min)
        + "--latitude-max {} ".format(latitude_max)
        + '--date-min "{} 12:00:00" '.format(date_min)
        + '--date-max "{} 12:00:00" '.format(date_max)
        + "--depth-min 0.494 --depth-max 0.4941 "
        + "--variable chl --variable dissic --variable fe "
        + "--variable no3 --variable nppv --variable o2 "
        + "--variable ph --variable phyc --variable po4 "
        + "--variable si --variable spco2 --variable talk "
        + "--out-dir {} ".format(filepath)
        + "--out-name {} ".format(filename)
        + "--user {} --pwd {}".format(username, password)
    )


def open_surphys(
    filepath="",
    date_min=None,
    date_max=None,
    latitude_min=-90,
    latitude_max=90,
    longitude_min=-180,
    longitude_max=180,
    username=None,
    password=None,
):
    """Open a CMEMS data file from the GLOBAL_ANALYSIS_FORECAST_PHY_001_024 dataset,
    downloading it first if it's not already available in the provided filepath.
    Output dataset includes surface fields of practical salinity (salinity), potential
    temperature in °C (theta), mixed layer depth in m (mld), sea surface height in m
    (ssh), eastwards (current_east) and northwards (current_north) current velocities in
    m/s, and total current speed in m/s (current_speed).

    Parameters
    ----------
    filepath : str, optional
        File path for the saved file, by default "".
    date_min : str, optional
        First date of data to download in '%Y-%m-%d' format, by default None, in which
        case today's date is used.
    date_max : str, optional
        Last date of data to download in '%Y-%m-%d' format, by default None, in which
        case today's date is used.
    latitude_min : int, optional
        Minimum latitude in decimal degrees N, by default -90.
    latitude_max : int, optional
        Maximum latitude in decimal degrees N, by default 90.
    longitude_min : int, optional
        Minimum longitude in decimal degrees E, by default -180.
    longitude_max : int, optional
        Maximum longitude in decimal degrees E, by default 180.
    username : str, optional
        Your CMEMS username, by default None, in which case you will be prompted for it.
    password : str, optional
        Your CMEMS password, by default None, in which case you will be prompted for it.

    Returns
    -------
    cmems
    """
    filename = get_surphys_filename(
        date_min, date_max, longitude_min, longitude_max, latitude_min, latitude_max
    )
    try:
        cmems = xr.open_dataset(filepath + filename)
    except FileNotFoundError:
        download_surphys(
            filename=filename,
            filepath=filepath,
            date_min=date_min,
            date_max=date_max,
            latitude_min=latitude_min,
            latitude_max=latitude_max,
            longitude_min=longitude_min,
            longitude_max=longitude_max,
            username=username,
            password=password,
        )
        cmems = xr.open_dataset(filepath + filename)
    cmems["current_speed"] = np.sqrt(cmems.uo**2 + cmems.vo**2)
    cmems = cmems.rename(
        {
            "mlotst": "mld",
            "so": "salinity",
            "thetao": "theta",
            "uo": "current_east",
            "vo": "current_north",
            "zos": "ssh",
        }
    )
    return cmems


def open_surbio(
    filepath="",
    date_min=None,
    date_max=None,
    latitude_min=-90,
    latitude_max=90,
    longitude_min=-180,
    longitude_max=180,
    username=None,
    password=None,
):
    """Open a CMEMS data file from the GLOBAL_ANALYSIS_FORECAST_BIO_001_028 dataset,
    downloading it first if it's not already available in the provided filepath.
    # Output dataset includes surface fields of practical salinity (salinity), potential
    # temperature in °C (theta), mixed layer depth in m (mld), sea surface height in m
    # (ssh), eastwards (current_east) and northwards (current_north) current velocities in
    # m/s, and total current speed in m/s (current_speed).

    Parameters
    ----------
    filepath : str, optional
        File path for the saved file, by default "".
    date_min : str, optional
        First date of data to download in '%Y-%m-%d' format, by default None, in which
        case today's date is used.
    date_max : str, optional
        Last date of data to download in '%Y-%m-%d' format, by default None, in which
        case today's date is used.
    latitude_min : int, optional
        Minimum latitude in decimal degrees N, by default -90.
    latitude_max : int, optional
        Maximum latitude in decimal degrees N, by default 90.
    longitude_min : int, optional
        Minimum longitude in decimal degrees E, by default -180.
    longitude_max : int, optional
        Maximum longitude in decimal degrees E, by default 180.
    username : str, optional
        Your CMEMS username, by default None, in which case you will be prompted for it.
    password : str, optional
        Your CMEMS password, by default None, in which case you will be prompted for it.

    Returns
    -------
    cmems
    """
    filename = get_surbio_filename(
        date_min, date_max, longitude_min, longitude_max, latitude_min, latitude_max
    )
    try:
        cmems = xr.open_dataset(filepath + filename)
    except FileNotFoundError:
        download_surbio(
            filename=filename,
            filepath=filepath,
            date_min=date_min,
            date_max=date_max,
            latitude_min=latitude_min,
            latitude_max=latitude_max,
            longitude_min=longitude_min,
            longitude_max=longitude_max,
            username=username,
            password=password,
        )
        cmems = xr.open_dataset(filepath + filename)
    return cmems

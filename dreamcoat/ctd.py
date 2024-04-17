import os
import textwrap
import pandas as pd
import numpy as np
from matplotlib import dates as mdates

ctd_renamer = {
    "scan": "scan",
    "prDM": "pressure",
    "t090C": "temperature",
    "CStarAt0": "beam_attenuation",
    "CStarTr0": "beam_transmission",
    "flC": "fluorescence_chelsea",
    "v2": "voltage_2",
    "ph": "pH_voltage",
    "sbeox0Mg/L": "oxygen_mgl",
    "sbeox0Mm/L": "oxygen_mml",
    "altM": "altimeter",
    "flECO-AFL": "fluorescence_wetlabs",
    "turbWETntu0": "turbidity",
    "timeS": "elapsed_seconds",
    "c0mS/cm": "conductivity",
    "sal00": "salinity",
    "potemp090C": "theta",
    "sigma-é00": "density",
    "depSM": "depth",
    "svCM": "sound_velocity",
    "bottle": "bottle",
    "c0S/m": "conductivity",
    "spar/sat/lin": "spar",
    "par/sat/log": "par",
    "sigma-t00": "sigma_theta",
}
ctd_Renamer = {k[0].upper() + k[1:]: v for k, v in ctd_renamer.items()}


def read_cnv_1Hz(filename):
    """Read the contents of a 1 Hz cnv file from a CTD cast.

    Parameters
    ----------

    filename : str
        The filename (and path) to read.

    Returns
    -------
    pd.DataFrame
        The imported dataset.
    """
    # Find number of header lines to skip and get column headers and other info
    with open(filename, "r", encoding="unicode_escape", errors="replace") as f:
        skiprows = 0
        names = []
        ctd_Hz_line = ""
        while not ctd_Hz_line.startswith("*END*"):
            ctd_Hz_line = f.readline()
            # Get column headers
            if ctd_Hz_line.startswith("# name "):
                names.append(ctd_Hz_line.split(" = ")[1].split(":")[0])
            # Get latitude and longitude
            if ctd_Hz_line.startswith("* NMEA Latitude = "):
                latitude_txt = ctd_Hz_line.split(" = ")[1].split(" ")
                latitude = float(latitude_txt[0]) + float(latitude_txt[1]) / 60
                if latitude_txt[2] == "S\n":
                    latitude *= -1
            if ctd_Hz_line.startswith("* NMEA Longitude = "):
                longitude_txt = ctd_Hz_line.split(" = ")[1].split(" ")
                longitude = float(longitude_txt[0]) + float(longitude_txt[1]) / 60
                if longitude_txt[2] == "W\n":
                    longitude *= -1
            # Get datetime information
            if ctd_Hz_line.startswith("# start_time = "):
                start_time = pd.to_datetime(ctd_Hz_line.split(" = ")[1].split("[")[0])
            # Increment row counter
            skiprows += 1
    # Deal with duplicate names by appending "_2" to their second appearance
    names_duplicate = []
    for i, name in enumerate(names):
        if names.count(name) == 2:
            names[names.index(name, i + 1)] += "_2"
            names_duplicate.append(names[names.index(name + "_2", i + 1)])
    # Import the dataset with pandas
    ctd_Hz = (
        pd.read_table(
            filename,
            encoding="unicode_escape",
            encoding_errors="replace",
            names=names,
            skiprows=skiprows,
            # delim_whitespace=True,
            sep=r"\s+",
        )
        .drop(columns=names_duplicate)
        .rename(columns=ctd_renamer)
    )
    # Add positional information to the dataframe
    ctd_Hz["datetime"] = start_time + np.arange(0, ctd_Hz.shape[0]).astype(
        "timedelta64[s]"
    )
    ctd_Hz["latitude"] = latitude
    ctd_Hz["longitude"] = longitude
    return ctd_Hz


def read_btl_raw(filename, colwidths=11, encoding="unicode_escape"):
    """Import a btl file from the CTD with minimal processing.

    Parameters
    ----------
    filename : str
        The btl filename (and path).
    colwidths : int, optional
        Width of each data column in the file, by default 11.
    encoding : str, optional
        Encoding of the .btl file, by default "unicode_escape".

    Returns
    -------
    pandas.DataFrame
        The btl file with minimal processing.
    dict
        Other potentially useful information.
    """
    btl_extras = {}
    # Find number of header lines to skip
    with open(filename, "r", encoding=encoding) as f:
        skiprows = -1
        btl_line = "*"
        while btl_line.startswith("*") or btl_line.startswith("#"):
            btl_line = f.readline()
            if btl_line.startswith("* NMEA Latitude"):
                btl_extras["latitude"] = btl_line.split(" = ")[1]
            elif btl_line.startswith("* NMEA Longitude"):
                btl_extras["longitude"] = btl_line.split(" = ")[1]
            skiprows += 1
    # This really awkward way of extracting the column headers is because sometimes they
    # run into each other (without whitespace in between) in the .btl files, which
    # prevents pandas from inferring the column widths correctly:
    headers = textwrap.wrap(
        btl_line.replace(" ", "_"), width=colwidths, break_on_hyphens=False
    )
    headers = [h.replace("_", "") for h in headers]
    headers = [ctd_Renamer[h] if h in ctd_Renamer else h for h in headers]
    headers.append("type")
    headers = {k: v for k, v in enumerate(headers)}
    # Now we can (hopefully) import the file
    btl_raw = pd.read_fwf(
        filename, skiprows=skiprows + 2, encoding=encoding, header=None
    ).rename(columns=headers)
    return btl_raw, btl_extras


def read_btl(filename, station=None, **kwargs):
    """Import a btl file from the CTD and format it nicely.

    Parameters
    ----------
    filename : str
        The btl filename (and path).
    station : str or int, optional
        The station number (which is added as a column), by default None.
    **kwargs
        Additional kwargs to pass to read_btl_raw().

    Returns
    -------
    pandas.DataFrame
        The nicely formatted bottle file.
    """
    # Read in the raw file
    btl, btl_extras = read_btl_raw(filename, **kwargs)
    # Parse datetime
    btl["date"] = btl.Date
    btl["time"] = btl.Date
    btl.loc[btl.bottle.isnull(), "date"] = np.nan
    btl.loc[~btl.bottle.isnull(), "time"] = np.nan
    btl["date"] = btl.date.ffill()
    btl["time"] = btl.time.bfill()
    btl["datetime"] = pd.to_datetime(btl.date + "T" + btl.time)
    btl["datenum"] = mdates.date2num(btl.datetime)
    btl.drop(columns=["Date", "date", "time"], inplace=True)
    # Put sdevs into the avg rows
    for c in ctd_renamer.values():  # first just copy the complete columns
        if c in btl and c != "bottle":
            btl[c + "_sd"] = btl[c]
    btl_avg = btl.type == "(avg)"
    btl_sdev = btl.type == "(sdev)"
    for c in ctd_renamer.values():
        if c in btl and c != "bottle":
            btl.loc[btl_sdev, c] = np.nan
            btl.loc[btl_avg, c + "_sd"] = np.nan
            btl[c + "_sd"] = btl[c + "_sd"].bfill()
    btl = btl[btl_avg].drop(columns="type").copy()
    # Parse longitude and latitude, if present in btl_extras
    if "latitude" in btl_extras:
        lat = btl_extras["latitude"].replace("\n", "")
        lat = lat.split()
        lat = " N".find(lat[2]) * float(lat[0]) + float(lat[1]) / 60
        btl_extras["latitude"] = lat
    if "longitude" in btl_extras:
        lon = btl_extras["longitude"].replace("\n", "")
        lon = lon.split()
        lon = " E".find(lon[2]) * float(lon[0]) + float(lon[1]) / 60
        btl_extras["longitude"] = lon
    # Add station and other extra columns
    btl["station"] = int(station)
    for k, v in btl_extras.items():
        btl[k] = v
    # Put data columns into alphabetical order
    btl_columns = btl.columns.sort_values()
    btl_starters = ["station", "bottle", *btl_extras.keys(), "datetime", "datenum"]
    btl_columns = [*btl_starters, *btl_columns.drop(btl_starters)]
    btl = btl[btl_columns]
    # Create station_bottle column and set as index
    btl["bottle"] = btl.bottle.astype(int)
    btl["station_bottle"] = btl.station.astype(str) + "-" + btl.bottle.astype(str)
    btl = btl.set_index("station_bottle")
    return btl


def read_btl_dir(btl_path, **kwargs):
    """Import all btl files in a directory into a single dataframe.

    Parameters
    ----------
    btl_path : str
        The path of the directory containing btl files.
    **kwargs
        Additional kwargs to pass to read_btl_raw().

    Returns
    -------
    pandas.DataFrame
        The nicely formatted bottle files.
    """
    # Get list of bottle files
    btl_files = os.listdir(btl_path)
    btl_files = [f for f in btl_files if f.endswith(".btl")]
    # Get corresponding station for each file
    btl_files = {int(f.split("-")[1].split("CTD")[0]): f for f in btl_files}
    btl_stations = list(btl_files.keys())
    # Import the bottle files in ascending station order and concatenate into a single
    # dataframe
    btl_stations.sort()
    btl = []
    for station in btl_stations:
        btl.append(read_btl(btl_path + btl_files[station], station=station, **kwargs))
    btl = pd.concat(btl)
    return btl


def read_bl(filename_bl):
    """Read the bottle numbers and datetime from a .bl file from the CTD.

    Parameters
    ----------
    filename_bl : str
        The filename (and path) for the .bl file.

    Returns
    -------
    pd.DataFrame
        The bottle numbers and datetime of bottle firing.
    """
    with open(filename_bl, "r") as f:
        bl = f.read().splitlines()
    skiprows = 0
    while not bl[skiprows].startswith("1, 1,"):
        skiprows += 1
    bl = pd.read_csv(
        filename_bl,
        skiprows=skiprows,
        header=None,
        names=["bottle", "x", "datetime", "y", "z"],
    ).drop(columns=["x", "y", "z"])
    bl["datetime"] = pd.to_datetime(bl.datetime)
    bl["datenum"] = mdates.date2num(bl.datetime)
    return bl


def read_ctd_bl_create_btl(filename_cnv_1Hz, filename_bl):
    ctd = read_cnv_1Hz(filename_cnv_1Hz)
    bl = read_bl(filename_bl)
    # Make new btl table
    btl = bl.set_index("bottle")
    # Add data from 30 seconds before firing for each sensor
    time_before_firing = pd.Timedelta("30s")
    for c in ctd.columns:
        if c not in btl:
            btl[c] = np.nan
            btl[c + "_std"] = np.nan
            btl[c + "_flag"] = np.nan
    ctd["bottle"] = 0
    for i, row in btl.iterrows():
        L = (ctd.datetime > row.datetime - time_before_firing) & (
            ctd.datetime <= row.datetime
        )
        ctd.loc[L, "bottle"] = i
        for c in ctd.columns:
            if c not in ["bottle", "datetime", "datenum"]:
                btl.loc[i, c] = ctd.loc[L, c].mean()
                btl.loc[i, c + "_std"] = ctd.loc[L, c].std()
    return ctd, btl


def read_ctd_create_btl(filename_cnv_1Hz, filename_btl_raw):
    ctd = read_cnv_1Hz(filename_cnv_1Hz)
    btl_raw = read_btl_raw(filename_btl_raw)
    # Make new btl table
    btl = btl_raw[~btl_raw.bottle.isnull()][
        ["bottle", "datetime", "datenum"]
    ].set_index("bottle")
    # Add data from 30 seconds before firing for each sensor
    time_before_firing = pd.Timedelta("30s")
    for c in ctd.columns:
        if c not in btl:
            btl[c] = np.nan
            btl[c + "_std"] = np.nan
            btl[c + "_flag"] = np.nan
    ctd["bottle"] = 0
    for i, row in btl.iterrows():
        L = (ctd.datetime > row.datetime - time_before_firing) & (
            ctd.datetime <= row.datetime
        )
        ctd.loc[L, "bottle"] = i
        for c in ctd.columns:
            if c not in ["bottle", "datetime", "datenum"]:
                btl.loc[i, c] = ctd.loc[L, c].mean()
                btl.loc[i, c + "_std"] = ctd.loc[L, c].std()
    return ctd, btl

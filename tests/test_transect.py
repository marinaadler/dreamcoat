import pandas as pd
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from scipy.interpolate import RBFInterpolator
import matplotlib as mpl
from matplotlib.font_manager import FontProperties
import dreamcoat as dc
import great_circle_calculator.great_circle_calculator as gcc


# %% Load data
bathy = xr.open_dataset("tests/data/nwes_emodnet_coarse.nc")
btl = pd.read_parquet("tests/data/nwes_btl.parquet")

#%% Define sections
# outflow = [1, 59, 112, 81, 86]
# inflow = [8, 46, 63, 93]
transect1 = [27, 24, 19, 8, 1]
# transect2 = [35, 40, 46, 50, 59]
# transect3 = [70, 63, 74, 112, 117]
# transect4 = [93, 89, 86, 98]
t_stations = transect1.copy()

#%% Get route distance for samples
# Create dataset with station number, lat and lon
stations = btl[["station", "latitude", "longitude"]].groupby("station").mean()

# Get lat and lon values for selected transect and distance values
stations.loc[t_stations, "distance"] = dc.maps.get_route_distance(
    np.array([stations.loc[t_stations].longitude.values, stations.loc[t_stations].latitude.values])
)
#Add distance values to btl for the selected stations in the transect
#Create transect_distance column and fill it with nan values
btl["distance"] = stations.loc[btl.station].distance.values

# Make with also extra start and end points
route_lon = stations.loc[t_stations].longitude.values
route_lat = stations.loc[t_stations].latitude.values

#%%
#########################Changes by Marina from here on ###############################

def extend_route(route_lon, route_lat, extra_fraction=0.05):
    """Add extra points at the start and end of a route.
    
    Parameters
    ----------
    route_lon : (n,) array-like
        The longitude points for the route in decimal degrees.
    route_lat : (n,) array-like
        The latitude points for the route in decimal degrees.
    extra_fraction : float, optional
        What fraction of the total route distance to extend by, by default 0.05.
        
    Returns
    -------
    route_lon_ext : (n + 2,) array-like
        The longitude points for the extended route in decimal degrees.
    route_lat_ext : (n + 2,) array-like
        The latitude points for the extended route in decimal degrees.
    route_distance_ext : (n + 2,) array-like
        The distances along the extended route in km from the first point of the
        original, unextended route.
    """
    #Calculate distances in the selected transect
    route_distance = dc.maps.get_route_distance(np.array([route_lon, route_lat]))
    #Select last value of the array which is the entire route distance
    total_distance = route_distance[-1]
    #Calculate which way the lines should be extended (extra stations)
    bearing_start = gcc.bearing_at_p1(
        (route_lon[1], route_lat[1]), (route_lon[0], route_lat[0])
    )
    bearing_end = gcc.bearing_at_p1(
        (route_lon[-2], route_lat[-2]), (route_lon[-1], route_lat[-1])
    )
    #Get lat and lon coordinates for the extra points
    extra_start = gcc.point_given_start_and_bearing(
        (route_lon[0], route_lat[0]),
        bearing_start,
        extra_fraction * total_distance * 1000,
    )
    extra_end = gcc.point_given_start_and_bearing(
        (route_lon[-1], route_lat[-1]),
        bearing_end,
        extra_fraction * total_distance * 1000,
    )
    #Add longitude and latitude values of the extra start and end points
    route_lon_ext = np.array([extra_start[0], *route_lon, extra_end[0]])
    route_lat_ext = np.array([extra_start[1], *route_lat, extra_end[1]])
    #Calculate new route distance
    route_distance_ext = dc.maps.get_route_distance(np.array([route_lon_ext, route_lat_ext]))
    #Shift the values so the first real station is 0
    route_distance_ext -= route_distance_ext[1]
    #Things that come out of the function
    return route_lon_ext, route_lat_ext, route_distance_ext

#%% Apply function
route_lon_ext, route_lat_ext, route_distance_ext = extend_route(route_lon, route_lat)
#%% Plot route to check
fig, ax = plt.subplots(dpi=300)
ax.scatter(route_lon, route_lat)
ax.scatter(route_lon_ext, route_lat_ext, marker='x')

# %%
f_settings = {
    'pb_final': dict(clabel='Pb [nmol L$^{-1}$]',
                      vmin=0,
                      vmax=0.075,
                      comap='viridis'),
   
}
#%% 
# Loop through f_settings to get the z variable and the settings for that variable
for zvar, settings in f_settings.items():
    L = np.isin(btl["station"], t_stations)
    x = btl[L]["distance"].values
    y = btl[L]["depth"].values
    z = btl[L][zvar].values

    #Select not nan values
    L = ~np.isnan(x) & ~np.isnan(y) & ~np.isnan(z)
    x = x[L]
    y = y[L]
    z = z[L]
    # x = x[::3]
    # y = y[::3]
    # z = z[::3]

    xscale = 0.9
    xy = np.array([x * xscale, y]).T  # transpose to get good shape

    interp = RBFInterpolator(xy, z, kernel="linear")

    # Create grid
    gx = np.linspace(np.min(x) + route_distance_ext[0], np.max(x) - route_distance_ext[0], num=200)
    gy = np.linspace(0, 450, num=200)
    gx, gy = np.meshgrid(gx, gy)

    # Predict salinity
    gz = interp(np.array([gx.ravel() * xscale, gy.ravel()]).T)

    # Create grid from predicted salinity values
    gz = np.reshape(gz, gx.shape)

    wp_interp = dc.maps.linspace_gc_waypoints(
        np.array([route_lon_ext, route_lat_ext]), num_approx=200
    )
    b_lon, b_lat = wp_interp
    b_distance = dc.maps.get_route_distance(wp_interp) + route_distance_ext[0]

    
    b_lon = xr.DataArray(b_lon, dims="b")
    b_lat = xr.DataArray(b_lat, dims="b")
    b_depth = -bathy.elevation.interp(lon=b_lon, lat=b_lat).data

    cmap = settings["comap"]

    fig, ax = plt.subplots(dpi=300)
    fc = ax.contourf(
        gx, gy, gz, 80, cmap=cmap, vmin=settings["vmin"], vmax=settings["vmax"]
    )

    # vmin=np.min(z),
    # vmax=np.max(z)

    ax.contour(
        gx,
        gy,
        gz,
        12,
        #[33, 34],
        colors="k",
        alpha=0.7,
        linewidths=0.7)

    ax.scatter(
        x,
        y,
        c=z,
        edgecolor="k",
        cmap=cmap,
        s=13,
        vmin=settings["vmin"],
        vmax=settings["vmax"],
        zorder=10,
    )

    ax.fill_between(b_distance, np.max(gy), b_depth, facecolor='xkcd:dark', zorder=9)
    ax.set_xlim((np.min(gx), np.max(gx)))
    ax.set_ylim((0, np.max(gy)))
    ax.set_xlabel("Distance [km]", family="serif")
    ax.set_ylabel("Depth [m]", family="serif")

    font_prop = FontProperties(family="serif")
    bounds = np.linspace(settings["vmin"], settings["vmax"], num=cmap)
    norm = mpl.colors.BoundaryNorm(boundaries=bounds, ncolors=cmap)
    cb = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=settings["comap"]),
        ax=ax,
        orientation="vertical",
        # ticks=[0.02, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.1, 2.4, 2.7]
    )

    # Set colorbar font
    font_prop = FontProperties(family="serif")
    cb.set_label(settings["clabel"])
    cb.ax.yaxis.label.set_font_properties(font_prop)
    plt.setp(cb.ax.get_yticklabels(), fontproperties=font_prop)

    # Get rid of minor ticks of the cb
    cb.ax.minorticks_off()
    # cb.ax.yaxis.set_ticks(np.arange(vmin, vmax, 0.5), minor=False)

    # plt.colorbar(fc, label="Fe [nmol kg$^{-1}$]")
    # cb = plt.colorbar(fc)
    # cb.set_label(label="Salinity [PSU]", family="serif")
    # ax.set_aspect(1)
    ax.invert_yaxis()
    plt.show()
    # #%%
    # bathy.elevation.coarsen(lon=100, lat=100, boundary='trim').mean().plot()

    # Fe [nmol L$^{-1}$]
    # Temperature [°C]

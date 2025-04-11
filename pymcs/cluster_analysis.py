from typing import Tuple
import xarray as xr
import numpy as np
import pandas as pd
import scipy.ndimage as ndimage

from pyorg.core.geometry import get_cells_area
from pyorg.core.convection import convective_regions
from pyorg.core.clusters import get_clusters, get_clusters_areas, \
    get_clusters_centroids, get_clusters_near_neighbor_distance, \
    _relabel_clusters

from analysis_functionality import EARTH_RADIUS, TARGET_LATLON, DATA_FREQUENCY

STEPHAN_BOLTZMANN_CONSTANT = 5.670374419*10**(-8)
EARTH_RADIUS_KM = EARTH_RADIUS/1000. #km

# ------------------------------------------------------------------------------
# Functions to build and subsample clusters
# ------------------------------------------------------------------------------
def calc_brightness_temperature(olr: xr.DataArray) -> xr.DataArray:
    """
    Calculate the brightness temperature from outgoing longwave radiation (OLR).

    This function computes the brightness temperature using the Stefan-Boltzmann 
    law, assuming the OLR is proportional to the fourth power of temperature.

    Parameters
    ----------
    olr : xr.DataArray
        Outgoing longwave radiation (OLR) in W/m^2.

    Returns
    -------
    xr.DataArray
        Brightness temperature in Kelvin (K), calculated as
        T_bright = (OLR / sigma)^(1/4), where sigma is the Stefan-Boltzmann
        constant.
    """
    T_bright = (olr/STEPHAN_BOLTZMANN_CONSTANT)**(1./4.)
    return T_bright


def calc_cluster_quantities(
        data: xr.DataArray,
        threshold: float,
        ) -> xr.DataArray:
    # Initialisation
    clusters_list, centroids_list, radiuses_list, areas_list, distances_list, \
        times_list = ([] for _ in range(6))
    total_clusters = 0
    cells_area = get_cells_area(data)
    
    for time in data.time:
        # calculate stuff for a single timestep
        threshold, regions = convective_regions(
            data.sel(time=time), threshold=threshold
            )
        clusters, clusters_number = get_clusters(
            regions,
            periodic_longitude_clustering=True, remove_edge_clusters=True,
            )
        clusters_index = list(range(1, clusters_number + 1))
        clusters_centroids = get_clusters_centroids(
            clusters, clusters_index, regions
            )
        clusters_distance = get_clusters_near_neighbor_distance(
            clusters_centroids
            )
        clusters_distance = [i/1000. for i in clusters_distance] #km

        clusters_areas = get_clusters_areas(
            clusters, clusters_index, cells_area
            )
        clusters_radiuses = np.sqrt(clusters_areas / np.pi) / 1000  # km
        clusters_areas = clusters_areas / 1000**2  # km^2

        clusters += total_clusters
        clusters = clusters.where(clusters != total_clusters, 0)
        clusters_list.append(clusters)
        total_clusters += clusters_number
        
        # append that stuff to list
        areas_list = areas_list + clusters_areas.tolist()
        centroids_list = centroids_list + clusters_centroids
        radiuses_list = radiuses_list + clusters_radiuses.tolist()
        distances_list = distances_list + clusters_distance
        times_list = times_list + [time.values]*len(clusters_radiuses)

    clusters = xr.concat(clusters_list, "time") 

    locations = pd.DataFrame(
        centroids_list, columns=["lat", "lon"],
        index=(list(range(1, total_clusters + 1)))
        )
    locations["radius_km"] = radiuses_list
    locations["areas_km2"] = areas_list
    locations["clusters_NN"] = distances_list
    locations["time"] = times_list
    locations.index.name = "cluster_id"

    return clusters, locations


def filter_clusters_triggering(
        clusters: xr.DataArray,
        locations: pd.DataFrame,
        radius_km: float
        ) -> pd.DataFrame:
    filtered_clusters = []
    for cluster_id, row in locations.iterrows():
        # Select clusters with no clusters in trigger region @ previous timestep
        region = _sample_area_of_interest(
            clusters, row, radius_km, lag_integer=1
            )
        if region is not None and (region == 0).all():
            filtered_clusters.append(cluster_id)
    
    # Filter the locations DataFrame to retain only the filtered clusters
    locations = locations.loc[filtered_clusters]
    #locations = _relabel_locations(locations)
    return locations



def _sample_area_of_interest(
        clusters: xr.DataArray,
        row: pd.Series,
        radius_km: float,
        lag_integer: int = 1
        ) -> None:
    
    # Get the cluster's location
    lat, lon, time = row['lat'], row['lon'], row['time']
    
    # Get the previous timestep and skip if it is not available
    prev_time = time - lag_integer*DATA_FREQUENCY
    if prev_time not in clusters.time.values:
        return
    
    # Extract the cluster's region and surrounding area at previous timestep
    cluster_prev_time = clusters.sel(time=prev_time)
    
    # Define the region within the given distance
    radius_kmeter = row['radius_km'] + radius_km
    radius_lat = radius_kmeter / \
        (EARTH_RADIUS_KM * 2*np.pi/360)  # Approx. conversion from km to deg
    radius_lon = radius_kmeter / \
        (EARTH_RADIUS_KM * 2*np.pi/360 * np.cos(np.deg2rad(lat)))
    region = cluster_prev_time.sel(
        lat=slice(max(TARGET_LATLON['lats'][0], lat-radius_lat),
                  min(TARGET_LATLON['lats'][1], lat+radius_lat)
                  ),
        lon=slice(max(TARGET_LATLON['lons'][0], lon-radius_lon),
                  min(TARGET_LATLON['lons'][1], lon+radius_lon)
                  )
        )
    return region


def subsample_clusters_regionally(
        clusters: xr.DataArray,
        locations: pd.DataFrame,
        valid_mask: xr.DataArray,
        remove_edgecluster: bool=True,
        ) -> Tuple[xr.DataArray, pd.DataFrame]:
    """
    Subsample clusters regionally by applying a valid mask and optionally
    removing edge clusters.

    Parameters
    ----------
    clusters : xr.DataArray
        A labeled array where each unique integer represents a cluster.
    valid_mask : xr.DataArray
        A boolean mask indicating the valid regions for subsampling.
    remove_edgecluster : bool, optional
        If True, clusters touching the edges of the valid mask will be removed. 
        Default is True.

    Returns
    -------
    xr.DataArray
        The subsampled clusters with invalid regions masked out and optionally 
        edge clusters removed.
    """
    clusters = clusters.where(valid_mask)
    if remove_edgecluster:
        clusters = _remove_edge_clusters(clusters, valid_mask)
    locations = select_locations_by_clusters(locations, clusters)
    #clusters, locations = _relabel(clusters, locations)
    return clusters, locations


def _remove_edge_clusters(
        clusters: xr.DataArray,
        valid_mask: xr.DataArray,
        ) -> xr.DataArray:
    """
    Remove clusters that are located at the edges of the valid data mask.

    This function identifies clusters that are direct neighbors of invalid
    (NaN) regions in the data and removes them by setting their values to 0.

    Parameters
    ----------
    clusters : xr.DataArray
        A labeled array where each unique integer represents a cluster.
    valid_mask : xr.DataArray
        A boolean mask indicating valid (True) and invalid (False) regions
        in the data.

    Returns
    -------
    xr.DataArray
        A modified version of the input `clusters` array where clusters
        neighboring invalid regions have been removed.
    """ 
    # Identify clusters that are direct neighbors of nan
    nan_mask = ~valid_mask
    neighbor_mask = ndimage.binary_dilation(
        nan_mask, structure=_2d_spatial_structure_element(),
        )

    # Mask clusters that are direct neighbors of NaN
    edge_clusters = clusters.where(neighbor_mask & (clusters > 0))
    edge_cluster_ids = _get_unique_cluster_ids(edge_clusters)
    return clusters.where(~clusters.isin(edge_cluster_ids), other=0)


def _2d_spatial_structure_element() -> np.ndarray:
    """
    Creates a 2D spatial structure element for morphological operations.

    The structure element is a 3D array where the middle slice allows dilation 
    in the x and y directions, while the top and bottom slices prevent dilation 
    in the z direction.

    Returns
    -------
    np.ndarray
        A 3D numpy array representing the structure element.
    """
    structure = np.array([[[0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]],  # No dilation in z direction
                          [[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]],  # Dilation in x and y
                          [[0, 0, 0],
                           [0, 0, 0],
                           [0, 0, 0]]]  # No dilation in z direction))
                        )
    return structure


def select_clusters_by_locations(
    clusters: xr.DataArray,
    locations: pd.DataFrame,
    ) -> xr.DataArray:
    cluster_ids = locations.index.values
    return clusters.where(clusters.isin(cluster_ids) | clusters.isnull(), 0)


def select_locations_by_clusters(
        locations: pd.DataFrame,
        clusters: xr.DataArray,
        ) -> pd.DataFrame:
    unique_clusters = _get_unique_cluster_ids(clusters)
    return locations.loc[locations.index.isin(unique_clusters)]


def _get_unique_cluster_ids(clusters: xr.DataArray) -> np.ndarray:
    cluster_ids = np.unique(clusters.values)
    cluster_ids = cluster_ids[~np.isnan(cluster_ids)]
    return cluster_ids


# ------------------------------------------------------------------------------
# Functions to build and subsample clusters
# ------------------------------------------------------------------------------
def get_pre_cluster_convergence(
        clusters: xr.DataArray,
        locations: pd.DataFrame,
        radius_km: float
        ) -> pd.DataFrame:
    _get_pre_cluster_quantity()


def _get_pre_cluster_quantity(
        clusters: xr.DataArray,
        locations: pd.DataFrame,
        radius_km: float
        ) -> pd.DataFrame:
    for cluster_id in locations.index:
        # Get the cluster's location
        lat, lon, time = locations.loc[cluster_id, ['lat', 'lon', 'time']]
        
        # Get the previous timestep and skip if it is not available
        prev_time = time - DATA_FREQUENCY
        if prev_time not in clusters.time.values:
            continue

    
def _select_area_of_interest():
    # Define the area of interest
    lat_min, lat_max = TARGET_LATLON['lats']
    lon_min, lon_max = TARGET_LATLON['lons']

    # Create a mask for the area of interest
    area_mask = (clusters.lat >= lat_min) & (clusters.lat <= lat_max) & \
                (clusters.lon >= lon_min) & (clusters.lon <= lon_max)
   
    return area_mask


# ------------------------------------------------------------------------------
# Functions that are currently unused
# ------------------------------------------------------------------------------
def _relabel(
        clusters: xr.DataArray,
        locations: pd.DataFrame
        ) -> Tuple[xr.DataArray, pd.DataFrame]:
    clusters.data = _relabel_clusters(clusters)
    locations = _relabel_locations(locations)
    return clusters, locations


def _relabel_locations(locations: pd.DataFrame) -> pd.DataFrame:
    locations = locations.reset_index()
    locations['cluster_id'] = range(1, len(locations) + 1)
    locations = locations.set_index('cluster_id')
    return locations
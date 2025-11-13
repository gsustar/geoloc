import os
import math
import numpy as np
from pyproj import Transformer

DEBUG = int(os.getenv("DEBUG", 0))

WGS84_A = 6378137.0  # semi-major axis in meters
WGS84_B = 6356752.314245  # semi-minor axis in meters
WGS84_E2 = 1 - (WGS84_B**2 / WGS84_A**2)  # eccentricity squared


def crs_transform(point: tuple, src_crs: str, tgt_crs: str):
    transformer = Transformer.from_crs(src_crs, tgt_crs, always_xy=True)
    return transformer.transform(point[0], point[1])


def geodetic_to_ecef(lat_deg, lon_deg, h):
    """
    Convert geodetic coordinates (lat, lon, alt) to ECEF (x, y, z)
    Returns:
            x, y, z in meters
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    N = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(lat) ** 2)
    X = (N + h) * math.cos(lat) * math.cos(lon)
    Y = (N + h) * math.cos(lat) * math.sin(lon)
    Z = (N * (1 - WGS84_E2) + h) * math.sin(lat)
    return X, Y, Z


def ecef_to_geodetic(x, y, z):
    """
    Convert ECEF (x, y, z) to geodetic coordinates (lat, lon, alt)
    Returns:
            lat, lon in degrees
            alt in meters
    """
    # Longitude
    lon = np.arctan2(y, x)

    # Iterative computation for latitude
    r = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, r * (1 - WGS84_E2))  # initial guess
    lat_prev = 0
    while np.abs(lat - lat_prev) > 1e-12:
        lat_prev = lat
        N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
        alt = r / np.cos(lat) - N
        lat = np.arctan2(z, r * (1 - WGS84_E2 * N / (N + alt)))

    N = WGS84_A / np.sqrt(1 - WGS84_E2 * np.sin(lat) ** 2)
    alt = r / np.cos(lat) - N

    # Convert radians to degrees
    lat = np.degrees(lat)
    lon = np.degrees(lon)
    return lat, lon, alt


def color_text(text, color="red"):
    colors = {
        "red": "\033[91m",
        "yellow": "\033[93m",
        "green": "\033[92m",
        "blue": "\033[94m",
        "reset": "\033[0m",
    }
    return f"{colors.get(color, '')}{text}{colors['reset']}"

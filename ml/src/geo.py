"""Great-circle geometry on a spherical Earth (vectorised, NaN-propagating)."""
from __future__ import annotations

import numpy as np

EARTH_RADIUS_M = 6_371_008.8  # IUGG mean radius


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres between points given in degrees."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dlmb = np.radians(lon2) - np.radians(lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def bearing_deg(lat1, lon1, lat2, lon2):
    """Initial bearing in degrees clockwise from north, in [0, 360)."""
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dlmb = np.radians(lon2) - np.radians(lon1)
    y = np.sin(dlmb) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dlmb)
    return (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0


def destination(lat, lon, bearing, dist_m):
    """Point reached from (lat, lon) travelling dist_m along an initial bearing (degrees)."""
    p1, l1 = np.radians(lat), np.radians(lon)
    theta = np.radians(bearing)
    delta = np.asarray(dist_m) / EARTH_RADIUS_M
    p2 = np.arcsin(np.sin(p1) * np.cos(delta) + np.cos(p1) * np.sin(delta) * np.cos(theta))
    l2 = l1 + np.arctan2(
        np.sin(theta) * np.sin(delta) * np.cos(p1),
        np.cos(delta) - np.sin(p1) * np.sin(p2),
    )
    return np.degrees(p2), (np.degrees(l2) + 540.0) % 360.0 - 180.0


def wrap_angle_deg(angle):
    """Wrap an angle difference into [-180, 180)."""
    return (np.asarray(angle) + 180.0) % 360.0 - 180.0


def enu_to_latlon(lat0, lon0, east_m, north_m):
    """Small-area local tangent-plane offset (metres) to lat/lon. Used by the simulator only."""
    lat = lat0 + np.degrees(np.asarray(north_m) / EARTH_RADIUS_M)
    lon = lon0 + np.degrees(np.asarray(east_m) / (EARTH_RADIUS_M * np.cos(np.radians(lat0))))
    return lat, lon

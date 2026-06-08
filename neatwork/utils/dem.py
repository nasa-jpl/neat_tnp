"""Self-contained replacements for `gdaldem` utilities.

Each terrain attribute is expressed as a standard `scipy.ndimage` primitive,
matching gdaldem's definitions:

* slope (percent) -- Horn (1981); Horn's 3x3 kernel is exactly the Sobel kernel.
* roughness        -- max - min over the 3x3 window (Wilson et al. 2007).
* hillshade        -- Horn slope/aspect with GDAL's default az=315, alt=45.

gdaldem leaves a 1-pixel nodata border (no ``-compute_edges``) and emits nodata
wherever any cell in the 3x3 window is nodata; ``_mask`` reproduces both.
"""
import math
import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

def _mask(out: NDArray, z: NDArray) -> NDArray:
    """Apply gdaldem's nodata convention: NaN border + NaN if any window cell is NaN."""
    out[ndimage.binary_dilation(~np.isfinite(z), np.ones((3, 3), bool))] = np.nan
    out[0, :] = out[-1, :] = out[:, 0] = out[:, -1] = np.nan
    return out


def _slope_percent(z: NDArray, cellsize: float = 1.0) -> NDArray:
    gx = ndimage.sobel(z, axis=1) / (8.0 * cellsize)
    gy = ndimage.sobel(z, axis=0) / (8.0 * cellsize)
    return _mask(100.0 * np.hypot(gx, gy), z)


def _roughness(z: NDArray) -> NDArray:
    return _mask(ndimage.maximum_filter(z, 3) - ndimage.minimum_filter(z, 3), z)


def _hillshade(z: NDArray, cellsize: float = 1.0,
              azimuth: float = 315.0, altitude: float = 45.0) -> NDArray:
    fx = ndimage.sobel(z, axis=1) / (8.0 * cellsize)   # dz/dx
    fy = ndimage.sobel(z, axis=0) / (8.0 * cellsize)   # dz/dy
    zenith, az = np.deg2rad(90.0 - altitude), np.deg2rad(360.0 - azimuth + 90.0)
    slope, aspect = np.arctan(np.hypot(fx, fy)), np.arctan2(fy, -fx)
    hs = np.cos(zenith) * np.cos(slope) + np.sin(zenith) * np.sin(slope) * np.cos(az - aspect)
    return _mask(np.clip(255.0 * hs, 0.0, 255.0), z)


class DEM:
    """An elevation raster: float64 array (nodata -> NaN) plus pixel geometry."""
    def __init__(self, array, cellsize, transform=None):
        self.array = array
        self.cellsize = cellsize
        self.transform = transform   # rasterio Affine of this (sub)raster


def dem_open(path: str, upper_left_x=None, upper_left_y=None,
             lower_right_x=None, lower_right_y=None) -> DEM:
    """Read a DEM into memory, optionally cropping to a projected-coordinate window.

    The (upper_left_x, upper_left_y, lower_right_x, lower_right_y) window matches
    gdal.Translate(projWin=...) snapping; omit it only for rasters small enough to
    load whole.
    """
    import rasterio
    from rasterio.windows import Window
    with rasterio.open(path) as ds:
        win, transform = None, ds.transform
        if upper_left_x is not None:
            ox, dx, oy, dy = ds.transform.c, ds.transform.a, ds.transform.f, ds.transform.e
            c0, c1 = math.floor((upper_left_x - ox) / dx), math.floor((lower_right_x - ox) / dx)
            r0, r1 = math.floor((upper_left_y - oy) / dy), math.floor((lower_right_y - oy) / dy)
            win = Window(c0, r0, c1 - c0 + 1, r1 - r0 + 1)
            transform = ds.window_transform(win)
        arr = ds.read(1, window=win, masked=True).astype(np.float64).filled(np.nan)
        return DEM(arr, ds.res[0], transform)


def dem_steepness(dem: DEM) -> DEM:
    return DEM(_slope_percent(dem.array, dem.cellsize), dem.cellsize, dem.transform)


def dem_roughness(dem: DEM) -> DEM:
    return DEM(_roughness(dem.array), dem.cellsize, dem.transform)


def dem_hillshade(dem: DEM) -> DEM:
    return DEM(_hillshade(dem.array, dem.cellsize), dem.cellsize, dem.transform)

#
# cost-map construction (math unchanged from the gdal-based version)
#

def image_histogram_equalization(image: NDArray, number_bins: int = 256) -> NDArray:
    # see http://www.janeriksolem.net/histogram-equalization-with-python-and.html
    hist, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = hist.cumsum()
    cdf = (number_bins - 1) * cdf / cdf[-1]
    return np.interp(image.flatten(), bins[:-1], cdf).reshape(image.shape)


def downsample_2d(mat: NDArray, res) -> NDArray:
    from skimage.transform import resize
    return resize(mat, res)


def f_combine1(roughness: NDArray, steepness: NDArray, p: int = 2, res=None) -> NDArray:
    """Average the two normalized layers, equalize, then apply a contrast curve."""
    def custom_scale(x):
        offset = 0.05
        return (x ** p / (x ** p + (1 - x) ** p)) * (1 - offset) + offset

    cm = (roughness + steepness) / 2
    cm = image_histogram_equalization(cm) / 255.0
    cm = custom_scale(cm)
    return downsample_2d(cm, res) if res is not None else cm


def dem_to_cost_map(dem: DEM, f_combine=None, res=None, p: int = 2) -> NDArray:
    """Build a cost map from a DEM by deriving normalized roughness and slope
    layers and combining them (default: :func:`f_combine1`)."""
    rough = _roughness(dem.array)
    rough = np.nan_to_num(rough, nan=np.nanmax(rough))
    rough = (rough - rough.min()) / (rough.max() - rough.min())

    slope = _slope_percent(dem.array, dem.cellsize)
    slope = np.nan_to_num(slope, nan=np.nanmax(slope)) / 100.0  # percent -> fraction

    if f_combine is None:
        f_combine = lambda r, s: f_combine1(r, s, p=p, res=res)
    return f_combine(rough, slope)

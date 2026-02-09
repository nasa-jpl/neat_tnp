import numpy as np
from numpy.typing import NDArray
from numpy.lib.stride_tricks import sliding_window_view
from osgeo import gdal
from typing import List, Callable

def apply(funcs: List[Callable], ls: List):
    for f in funcs:
        ls = list(map(f, ls))
    return ls

gdal.UseExceptions()

# 
# operations on DEM data
# 

# convert to np.array but turn nodata into np.nan
def dem_to_ndarray(gdal_ds) -> np.ndarray:
    arr = gdal_ds.ReadAsArray()
    nodata = gdal_ds.GetRasterBand(1).GetNoDataValue()
    arr[arr == nodata] = np.nan
    return arr

def dem_crop(dem, upper_left_x, upper_left_y, lower_right_x, lower_right_y):
    return gdal.Translate(
        "", dem, format="MEM",
        projWin=[upper_left_x, upper_left_y, lower_right_x, lower_right_y]
    )

def dem_open(path: str) -> gdal.Dataset:
    return gdal.Open(path)

def dem_hillshade(dem: gdal.Dataset) -> gdal.Dataset:
    return gdal.DEMProcessing(
        "", dem, "hillshade",
        options=gdal.DEMProcessingOptions(format="MEM")
    )

def dem_roughness(dem: gdal.Dataset) -> gdal.Dataset:
    return gdal.DEMProcessing(
        "", dem, "roughness",
        options=gdal.DEMProcessingOptions(format="MEM")
    )

def dem_steepness(dem: gdal.Dataset) -> gdal.Dataset:
    return gdal.DEMProcessing(
        "", dem, "slope",
        options=gdal.DEMProcessingOptions(format="MEM")
    )

def normalize(mat: NDArray) -> NDArray:
    return (mat - mat.min()) / (mat.max() - mat.min())

def high_pass(mat: NDArray, threshold: float, low_val = 0.0) -> NDArray:
    mat[mat<=threshold] = low_val
    return mat

def cost_map_from(
        dem: gdal.Dataset, 
        subsampling_kernel_size: int = 1,
        histogram_equalization: bool = True,
        high_pass_thresh: float = 0.0,
        high_pass_low_val: float = 0.02,
        return_sequence: bool = False,
    ):
    cost_maps = [
        dem_roughness(dem),
        dem_steepness(dem),
    ]
    funcs = [
        dem_to_ndarray, 
        lambda x: downsample_2d(x, ksize=subsampling_kernel_size), 
        lambda mat: np.nan_to_num(mat, nan=np.nanmax(mat)),
        lambda mat: image_histogram_equalization(mat) if histogram_equalization else mat,
        normalize,
        lambda mat: high_pass(mat, threshold=high_pass_thresh, low_val=high_pass_low_val),
    ]
    
    if return_sequence:
        # Track intermediate results after each function application
        sequence = []
        current_maps = cost_maps
        for f in funcs:
            current_maps = list(map(f, current_maps))
            sequence.append(np.array(current_maps).mean(axis=0))
        return sequence
    else:
        cost_mats = apply(funcs, cost_maps)
        cost_mat = np.array(cost_mats).mean(axis=0)
        return cost_mat

# 
# operations on numpy ndarrays
# 

def conv2d_numpy(input_array: NDArray, kernel: NDArray, stride=1, padding=0, dilation=1) -> NDArray:
    if dilation != 1:
        raise NotImplementedError("dilation != 1 is not supported in the numpy implementation")

    input_h, input_w = input_array.shape
    kernel_h, kernel_w = kernel.shape

    # Step 1: Create valid mask (1 where not NaN, 0 where NaN)
    valid_mask = ~np.isnan(input_array)
    
    # Step 2: Replace NaNs with 0
    input_filled = np.nan_to_num(input_array, nan=0.0)

    # Apply padding
    if padding > 0:
        pad_width = ((padding, padding), (padding, padding))
        input_filled = np.pad(input_filled, pad_width, mode='constant', constant_values=0)
        valid_mask = np.pad(valid_mask, pad_width, mode='constant', constant_values=False)

    # Step 3: Create windows
    # sliding_window_view returns shape (out_h, out_w, k_h, k_w)
    # We slice to apply stride
    windows_input = sliding_window_view(input_filled, (kernel_h, kernel_w))[::stride, ::stride]
    # Cast mask to float32 for summing
    windows_mask = sliding_window_view(valid_mask.astype(np.float32), (kernel_h, kernel_w))[::stride, ::stride]

    # Step 4: Convolve input and mask
    # Perform element-wise multiplication and sum over the kernel dimensions (last two axes)
    conv_output = np.einsum('ijkl,kl->ij', windows_input, kernel)
    
    # Mask output (sum of valid pixels in window) to emulate convolution with ones
    mask_output = np.sum(windows_mask, axis=(-2, -1))

    # Step 5: Avoid divide-by-zero by setting minimum mask value
    eps = 1e-6
    normalized_output = conv_output / (mask_output + eps)

    return normalized_output

def downsample_2d(mat: NDArray, ksize=15):
    if ksize == 1:
        return mat
    kernel = np.full((ksize,ksize), 1/ksize**2)
    return conv2d_numpy(
        mat, kernel,
        stride=ksize//2, padding=0, dilation=1,
    )

def image_histogram_equalization(image: NDArray, number_bins=256) -> NDArray:
    # see http://www.janeriksolem.net/histogram-equalization-with-python-and.html

    # get pixel intensity histogram
    image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    cdf = image_histogram.cumsum() # cumulative distribution function
    cdf = (number_bins-1) * cdf / cdf[-1] # normalize

    # use linear interpolation of cdf to find new pixel values
    image_equalized = np.interp(image.flatten(), bins[:-1], cdf)

    return image_equalized.reshape(image.shape) #, cdf

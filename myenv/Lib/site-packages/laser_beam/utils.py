"""
Docstring for laser_beam.utils
"""
import xarray as xr
import numpy as np
import pint

import matplotlib.pyplot as plt
from matplotlib.path import Path

# Define the public API of the module (what gets imported with *)
__all__ = [
    'hello_world', 
    'info', 
    'gauss_1D',
    'gauss_2D', 'supergauss_round_2D', 'supergauss_square_2D', 'square_2D', 'ellipse_2D', 'squircle_2D', 'serrated_aperture_points',
    'rescale_by_units', 'convert_coord',
    'convert_wavelength_2_frequency',
    'pixel_size',
    'set_statistics',
    'get_units',
    ]

# set up unit registry
ureg = pint.UnitRegistry()
ureg.define('px = [index] = pixel')


def hello_world():
    print("Hello from Laser Beam!")

def template(arg_1:float, arg_2:str='Label') -> float:
    """
    Short summary line.

    Longer description if needed.

    Parameters
    ----------
    arg_1 : float
        Explanation of arg_1.

    arg_2 : str, default='Label'
        Explanation of arg_2.

    Returns
    -------
    output : float
        Explanation of the returned value.
    """
    output = arg_1 * 1
    return output

# 1D distributions:

# gauss:
def gauss_1D(x, amplitude=1.0, x0=0.0, width=10.0) -> np.ndarray:
    """
    Generate 1D Gaussian distribution.
    
    Parameters:
    - x: 1D array of x coordinates
    - amplitude: Peak amplitude
    - x0: Center position
    - width: width at 1/e^2 intensity (D4sigma)
    
    Returns:
    - 1D array of Gaussian values
    """
    s = width / 4  # convert D4sigma to standard deviation
    return amplitude * np.exp(-((x-x0) / np.sqrt(2) / s) ** 2)

# hyperbola:
def hyperbola_1D(x, w0=1.0, x0=0.0, slope= 1.0) -> np.ndarray:
    """
    Generate 1D hyperbolic distribution.
    
    Parameters:
    - x: 1D array of x coordinates
    - w0: minimum width at focus (x0)
    - x0: center position
    - slope: slope of the hyperbola
    
    Returns:
    - 1D array of hyperbolic values
    """
    z = x - x0
    return w0 * np.sqrt(1 + (z * slope / w0) ** 2)



# 2D distributions:
def gauss_2D(X, Y, amplitude=1.0, x0=0.0, y0=0.0, width_x=10.0, width_y=10.0, orientation=0, inverted = False) -> np.ndarray:
    """
    Generate 2D Gaussian beam profile.
    
    Parameters:
    - X, Y: 2D meshgrid arrays
    - amplitude: Peak amplitude
    - x0, y0: Center positions
    - width_x, width_y: width at 1/e^2 intensity (D4sigma)
    - orientation: rotate clockwise in deg
    - inverted: amplitude->0 & 0->amplitude
    
    Returns:
    - 2D array of Gaussian values
    """

    # rotate meshgrid
    X, Y = rotate_meshgrid_XY(X,Y,x0,y0,orientation)


    # convert D4sigma (1/e^2) to standard deviation
    s_x = width_x / 4
    s_y = width_y / 4

    data = amplitude * np.exp(-
        ((X-x0) / np.sqrt(2) / s_x) ** 2 -
        ((Y-y0) / np.sqrt(2) / s_y) ** 2
    )

    # invert data
    if inverted == True:
        data = amplitude - data

    return data

def supergauss_square_2D(X, Y, amplitude=1.0, x0=0.0, y0=0.0, width_x=1.0, width_y=1.0, order=4, orientation=0, inverted = False) -> np.ndarray:
    """
    Generate 2D Super-Gaussian beam profile (square).
    
    Parameters:
    - X, Y: 2D meshgrid arrays
    - amplitude: Peak amplitude
    - x0, y0: Center positions
    - width_x, width_y: Full width at half maximum (FWHM)
    - order: Order of the super-Gaussian
    - orientation: rotate clockwise in deg
    - inverted: amplitude->0 & 0->amplitude
    
    Returns:
    - 2D array of Super-Gaussian values
    """
    
    # rotate meshgrid
    X, Y = rotate_meshgrid_XY(X,Y,x0,y0,orientation)

    n = order
    scaling_factor = 2 * (np.log(2))**(1/(2*n))
    w_x = width_x / scaling_factor
    w_y = width_y / scaling_factor
    data = amplitude * np.exp(
        - ((X-x0) / w_x)**(2*n) -
        ((Y-y0) / w_y)**(2*n)
    )

    # invert data
    if inverted == True:
        data = amplitude - data

    return data

def supergauss_round_2D(X, Y, amplitude=1.0, x0=0.0, y0=0.0, width_x=1.0, width_y=1.0, order=4, orientation=0, inverted = False) -> np.ndarray:
    """
    Generate 2D Super-Gaussian beam profile (round).
    
    Parameters:
    - X, Y: 2D meshgrid arrays
    - amplitude: Peak amplitude
    - x0, y0: Center positions
    - width_x, width_y: Full width at half maximum (FWHM)
    - order: Order of the super-Gaussian
    - orientation: rotate clockwise in deg
    - inverted: amplitude->0 & 0->amplitude
    
    Returns:
    - 2D array of Super-Gaussian values
    """

    # rotate meshgrid
    X, Y = rotate_meshgrid_XY(X,Y,x0,y0,orientation)

    n = order
    scaling_factor = 2 * (np.log(2))**(1/(2*n))
    w_x = width_x / scaling_factor
    w_y = width_y / scaling_factor
    data = amplitude * np.exp(
        - (((X-x0) / w_x)**2 +
        ((Y-y0) / w_y)**2)**n
    )

    # invert data
    if inverted == True:
        data = amplitude - data

    return data

def square_2D(X, Y, amplitude=1.0, x0=0.0, y0=0.0, width_x=1.0, width_y=1.0, orientation=0, inverted = False):
    """
    Docstring for square_2D
    
    :param X: Description
    :param Y: Description
    :param amplitude: Description
    :param x0: Description
    :param y0: Description
    :param width_x: Description
    :param width_y: Description
    :param orientation: Description
    :inverted: amplitude->0 & 0->amplitude
    """
    # rotate meshgrid
    X, Y = rotate_meshgrid_XY(X,Y,x0,y0,orientation)

    data = np.where(
        (np.abs(X - x0) <= 0.5 * width_x) & (np.abs(Y - y0) <= 0.5 * width_y),
        amplitude,  # Inside the square, intensity = amplitude
        0           # Outside the square, intensity = 0
    )

    # invert data
    if inverted == True:
        data = amplitude - data

    return data

def ellipse_2D(X, Y, amplitude=1.0, x0=0.0, y0=0.0, width_x=1.0, width_y=1.0, orientation=0, inverted = False):
    """
    Docstring for square_2D
    
    :param X: Description
    :param Y: Description
    :param amplitude: Description
    :param x0: Description
    :param y0: Description
    :param width_x: Description
    :param width_y: Description
    :param orientation: Description
    :inverted: amplitude->0 & 0->amplitude
    """
    # rotate meshgrid
    X, Y = rotate_meshgrid_XY(X,Y,x0,y0,orientation)

    # check if (x,y) lies inside or outside of ellipse
    data = np.where(
        ((X - x0) ** 2 / (0.5 * width_x) ** 2) + ((Y - y0) ** 2 / (0.5 * width_y) ** 2) <= 1,
        amplitude,  # Inside the square, intensity = A
        0   # Outside the square, intensity = 0
    )

    # invert data
    if inverted == True:
        data = amplitude - data
        
    return data

def squircle_2D(X, Y,  amplitude=1.0, x0=0.0, y0=0.0, width_x=1.0, width_y=1.0, radius = 0.1, orientation=0, inverted = False):
    """
    Generate a 2D squircle (rectangle with rounded corners).

    :param X: 2D array of x coordinates
    :param Y: 2D array of y coordinates
    :param amplitude: value inside the squircle (default 1.0)
    :param x0: x-coordinate of squircle center (default 0.0)
    :param y0: y-coordinate of squircle center (default 0.0)
    :param width_x: total width of the rectangle (default 1.0)
    :param width_y: total height of the rectangle (default 1.0)
    :param radius: radius of the rounded corners (fillet, default 0.1)
    :param orientation: rotation angle in radians (default 0)
    :param inverted: if True, invert amplitude (default False)
    :return: 2D array of same shape as X/Y with amplitude inside the squircle
    """

    R = radius

    # rotate meshgrid
    X, Y = rotate_meshgrid_XY(X,Y,x0,y0,orientation)

    # distance to nearest rectangle edge (clamped)
    dx = np.clip(np.abs(X) - (width_x/2 - R), 0, None)
    dy = np.clip(np.abs(Y) - (width_y/2 - R), 0, None)

    # inside squircle if inside rectangle body or inside corner radius
    mask = dx**2 + dy**2 <= R**2

    data  = mask.astype(float) * amplitude

    # invert data
    if inverted == True:
        data = amplitude - data

    return data

def serrated_aperture(X, Y, amplitude=1.0, x0=0.0, y0=0.0, 
                      size=80, radius=10, depth=5, n_edge=10, n_corner=3, 
                      orientation=0.0, inverted=False):
    
    # rotate meshgrid
    X, Y = rotate_meshgrid_XY(X,Y,x0,y0,orientation)

    # 2. Generate the points in the local frame (centered at 0,0)
    # Note: size, r, and depth are relative to the local frame
    points = serrated_aperture_points(size=size, n_e=n_edge, n_c=n_corner, r=radius, depth=depth)

    # 3. Shift the points to the world center (x0, y0)
    # This aligns the (0,0) of the aperture with the (x0, y0) of the rotated grid
    points[:, 0] += x0
    points[:, 1] += y0

    # 4. Generate the Mask
    # Flatten the rotated grid for the path algorithm
    grid_points = np.column_stack((X.ravel(), Y.ravel()))
    
    path = Path(points)
    mask = path.contains_points(grid_points).reshape(X.shape)

    # 5. Apply Amplitude and Inversion
    data = mask.astype(float) * amplitude
    
    if inverted:
        data = amplitude - data

    return data

def serrated_aperture_points(size=5, r=5, depth=0.1, n_e=10, n_c=10):
    """
    Generates a set of (x, y) coordinates defining a serrated square aperture
    with rounded corners.
    
    The shape is constructed by defining one quadrant (one edge and one corner)
    and rotating it four times. The serrations are created by alternating 
    offsets relative to the nominal boundary.

    Parameters
    ----------
    size : float
        The side length of the square aperture (edge to edge).
    n_e : int
        Number of serration "teeth" along each straight edge. 
        Will be cast to an integer.
    n_c : int
        Number of serration "teeth" along each 90-degree corner arc.
        Will be cast to an integer.
    r : float
        The nominal radius of the corners. If r=0, corners are sharp.
        If r=size/2, the aperture becomes a serrated circle.
    depth : float
        The peak-to-peak depth of the serrations. A positive value 
        creates teeth alternating between +/- depth/2.

    Returns
    -------
    points : ndarray
        An (N, 2) array of coordinates defining the closed path of the 
        aperture, suitable for use with matplotlib.path or cv2.fillPoly.

    Raises
    ------
    ValueError
        If n_e or n_c are negative.
    """

    # some validation
    if n_e < 0:
        raise ValueError(f"Number of points (n_e={n_e}) must be non-negative.")
    if n_c < 0:
        raise ValueError(f"Number of points (n_c={n_c}) must be non-negative.")
    
    # Cast to int to ensure array sizing works
    n_e, n_c = int(n_e), int(n_c)


    # 1. Create the alternating "serration" offsets
    # Using tile to create [depth/2, -depth/2, depth/2, ...]
    edge_offsets = np.tile([depth/2, -depth/2], n_e + 1)[:2 * n_e + 1]
    corner_offsets = np.tile([-depth/2, depth/2], n_c + 1)[:2 * n_c + 1]

    # 2. Build the bottom edge (Edge 1)
    x_e = np.linspace(-(size/2 - r), (size/2 - r), 2 * n_e + 1)
    edge_template = np.column_stack((x_e, -size/2 + edge_offsets))

    # 3. Build the bottom-right corner (Corner 1)
    # Angle spans 0 to 90 degrees (pi/2)
    angles = np.linspace(0, np.pi/2, 2 * n_c + 1)
    r_coords = r + corner_offsets
    
    # Center of the corner arc is at (size/2 - r, -size/2 + r)
    cx, cy = (size/2 - r), -(size/2 - r)
    corner_template = np.column_stack((
        cx + r_coords * np.sin(angles),
        cy - r_coords * np.cos(angles)
    ))

    # 4. Combine into one side (Quadrant)
    side = np.vstack((edge_template, corner_template))

    # 5. Rotate and collect
    # 90-degree rotation matrix
    m_rot = np.array([[0, 1], [-1, 0]])
    
    all_points = []
    current_side = side
    for _ in range(4):
        all_points.append(current_side)
        current_side = current_side @ m_rot
        
    return np.vstack(all_points)


def rotate_meshgrid_XY(X ,Y, x0, y0, rotation):
    """
    Docstring for rotate_meshgrid_XY
    
    :param X: meshgrid X
    :param Y: Meshgrid Y
    :param x0: point of rotation along x
    :param y0: point of rotation along y
    :param rotation: rotate in deg
    """
    rotation *= -1 # clockwise rotation
    rotation = rotation * np.pi / 180  # convert to radians
    
    # shift point of rotation
    Xc = X - x0
    Yc = Y - y0

    # rotate coordinates
    Xr = Xc * np.cos(rotation) + Yc * np.sin(rotation)
    Yr = -Xc * np.sin(rotation) + Yc * np.cos(rotation)
    
    # shift back
    X = Xr + x0
    Y = Yr + y0

    return X, Y

# helper function to define coordinate:
def define_1D_array(*, N=None, delta=None, center=None, start=None, stop=None):
    """
    Define coordinate array 
    """
    # Count which parameters are provided
    params = {'N': N, 'delta': delta, 'center': center, 'start': start, 'stop': stop}
    provided = {k: v for k, v in params.items() if v is not None}

    # Option 1: N, delta, center
    if set(provided.keys()) == {'N', 'delta', 'center'}:
        length = N * delta
        start = center - length / 2
        array = np.arange(N) * delta + start

    # Option 2: N, delta, start
    elif set(provided.keys()) == {'N', 'delta', 'start'}:
        array = np.arange(N) * delta + start

    # Option 3: start, stop, N
    elif set(provided.keys()) == {'start', 'stop', 'N'}:
        array = np.linspace(start, stop, N)

    # Option 4: start, stop, delta
    elif set(provided.keys()) == {'start', 'stop', 'delta'}:
        N_calc = int(np.floor((stop - start) / delta)) + 1
        array = np.arange(N_calc) * delta + start

    else:
        # Generate helpful error message
        valid_combinations = [
            "N, delta, center",
            "N, delta, start",
            "start, stop, N",
            "start, stop, delta"
        ]
        raise ValueError(
            f"Invalid parameter combination: {', '.join(provided.keys())}\n"
            f"Valid combinations are:\n  - " + "\n  - ".join(valid_combinations)
        )


    return array

# rescale data to new units
def rescale_by_units(data, unit_data, unit_target):
    """
    Convert a value or array from one unit to another using pint.
    Carefull: pint doesn't handle rad: 'rad' = 1

    Parameters
    ----------
    data : float or array-like
    unit_data : str
        The unit of the input data
    unit_target : str
        The target unit for conversion  
    """
    try:
        quantity = data * ureg(unit_data)
        converted = quantity.to(unit_target)
        data_new = converted.magnitude
        
        # Calculate conversion factor safely
        # We use 1 * unit_data to find the ratio independently of the 'data' values
        factor = ureg.convert(1, unit_data, unit_target)
        
        #print(f"Converting '{unit_data}' to '{unit_target}' with converting factor {factor}")
    
    except Exception as e:
        raise ValueError(f"Can't convert '{unit_data}' to '{unit_target}'") from e
    
    return data_new

# rescale coordinate to new units
def convert_coord(da, coord_name, target_unit):
    """
    Convert a coordinate to a target unit using its own metadata.

    Returns:
        da (updated or original)
    """
    if target_unit is None:
        return da

    coord = da.coords[coord_name]
    current_unit = coord.attrs.get("units")

    if current_unit is None:
        print(f"No unit found for {coord_name}, skipping conversion.")
        return da

    try:
        new_values = rescale_by_units(coord.values, current_unit, target_unit)

        # assign updated coordinate
        da = da.assign_coords({
            coord_name: (coord.dims, new_values)
        })

        # update metadata
        da.coords[coord_name].attrs["units"] = target_unit

        return da

    except Exception as e:
        print(f"Conversion failed for {coord_name}: {e}")
        return da

# helper function: get units (Todo: check if still used anywhere)
def get_units(da):
    return da.attrs.get("units")


# 
def convert_wavelength_2_frequency(wavelength):
    """
    Convert wavelength in units of 'm' to frequency in units of '1/s' or 'Hz'
    
    Parameters
    ----------
    wavelength : float, 
        wavelength in units of 'm'
    
    Returns
    -------
    frequency : float
        Frequency in units of '1/s' or 'Hz'
    
    """
    # calculate frequency
    c0 = 299792458.0
    frequency = c0/wavelength
    return frequency

def convert_frequency_2_wavelength(frequency):
    """
    Convert frequency in units of '1/s' or 'Hz' to wavelength in units of 'm
    
    Parameters
    ----------
    frequency : float
        Frequency in units of '1/s' or 'Hz'
    
    Returns
    -------
    wavelength : float, 
        wavelength in units of 'm'
    """
    # calculate frequency
    c0 = 299792458.0
    wavelength = c0/frequency
    return wavelength

def set_statistics(da: xr.DataArray) -> xr.DataArray:

    da_new = da.copy() # create a copy to avoid modifying the original data
    total_sum = da.sum(skipna=True)

    for coord_name in da.coords:
        # 1. Get the coordinate as a DataArray (not .values)
        coord = da[coord_name]
        
        # 2. Alignment: Ensure coord has the same dimensions as the data
        # This allows (data * coord) to work regardless of dimensionality
        weighted_coord = da * coord
        
        # 3. Calculate Center (Weighted Average)
        center = weighted_coord.sum(skipna=True) / total_sum
        
            # # 4. Calculate Spread (Weighted Standard Deviation)
            # # Formula, efficient: sqrt(mean(x^2) - mean(x)^2)
            # weighted_coord_sq = da * (coord**2)
            # mean_sq = weighted_coord_sq.sum() / total_sum
            # spread = np.sqrt(mean_sq - center**2)

        # 4. Calculate Spread (Weighted Standard Deviation)
        # Formula, conventional: sqrt(sum(w_i * (x_i - center)^2) / sum(w_i))
        deviation_sq = (coord - center)**2
        variance = (da * deviation_sq).sum(skipna=True) / total_sum
        spread = np.sqrt(variance)
        
        # 2. Add to the coordinate attributes
        da_new.coords[coord_name].attrs['center'] = center.item()       # .item() converts 0-dim DataArray to scalar
        da_new.coords[coord_name].attrs['spread'] = spread.item()

    return da_new


# helper function: get pixel size of 1D array, can bee coordinate
def pixel_size(array):
    """
    Return pixel size for a 1D xarray coordinate after checking uniformity.

    Parameters
    ----------
    coord : xr.DataArray or array-like
        1D coordinate

    Returns
    -------
    float
        Pixel size (signed)
    """
    # get values from array
    if isinstance(array, xr.DataArray): # coordinate ins DataArray is itself a DataArray
        values = array.values
    else:
        values = np.asarray(array)
    pass

    # check dimension
    if values.ndim != 1:
        raise ValueError("Coordinate must be 1D")
    
    # calc difference between all values 
    diffs = np.diff(values)

    if diffs.size == 0:
        raise ValueError("Coordinate must have at least two values")
    
    abs_diffs = np.abs(diffs)       # use abs in case of decending order
    size = abs_diffs[0]             # use first elemen

    # check that all pixel size are the same
    if not np.allclose(abs_diffs, size, rtol=1e-9, atol=0.0): # rel tollerance >0, absolute tollerance = 0, (femto seconds vs GW, atol meaning less)
        raise ValueError("Coordinate is not uniformly spaced")
    
    return size



# print info about the laser beam
def info(xa):
    print(f"laser beam:")
    print(f"  name: {xa.attrs['label']}")
    print("  variable:")
    print(f"    - {xa.name}[{xa.attrs['units']}]")

    # print coordinates
    print("  Coordinates:")
    for dim in xa.dims:
        print(f"    - {dim}[{xa.coords[dim].attrs['units']}], N={xa.coords[dim].size}")
    

if __name__ == "__main__":

    import laser_beam as lb # run __init_.py to import other modules and set up units
    import matplotlib.pyplot as plt
    #ureg = pint_xarray.unit_registry



    # test functions
    hello_world()

    # test unit conversion
        #W/cm²
    unit_in =  'W/cm²'
    unit_out =  'W/mm²'
    scale = lb.rescale_by_units(1, unit_in, unit_out)
    print(f"unitconversion: '{unit_in}'/'{unit_out}' = {scale}")

    # test wavelength conversion:
    wavelength_nm = 1030
    frequency = convert_wavelength_2_frequency(wavelength_nm*1e-9)
    print(f"Frequency of {wavelength_nm}nm: f={frequency:.3e}'Hz'")

    # Create a laser beam
    beam = lb.create_beam_xy(
        type="SuperGaussSquare",
        name="Test Gaussian Beam",
        func_params={
            'width_x': 3,
            'width_y': 3,
            'amplitude': 100,
        },
        axis_unit="mm",
        axis_pixelsize=0.1,
    )

    
    print(f"pixel size along x: {pixel_size(beam["x"])}{beam["x"].attrs['units']}")
    
    #fig, axes = plt.subplots(2,1, figsize=(6,8))

    lb.plot_2D(beam)
    # lb.plot_2D(beam,ax=axes[1])


    # print info
    info(beam)

    #beam.plot()

    #print(beam)
    plt.tight_layout()
    plt.show()
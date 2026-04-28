
import numpy as np
import xarray as xr

from laser_beam.utils import rescale_by_units, pixel_size, set_statistics
from laser_beam.visualize import add_rectangle_overlay

__all__ = [
    'crop', 'cross_section', 
    'beam_arithmetic',
    'set_Energy',
    ]

# crop laser beam along given dimensions and ranges
def crop(beam,
    label = None,
    overlay_add=True,
    overlay_label = None,
    calc_statistics=True,
    **kwargs):
    """
    Crop an xarray object along multiple dimensions with unit awareness.

    add overlay to source (optionally): used for visualizaion 
    
    Example: crop(ds, wavelength=(400, 700, 'nm'), x=(0, 5, 'cm'))
    """
    # make copy of beam for output
    beam_out = beam.copy()
    beam_out.attrs = dict(beam.attrs)

    # remove existing overlays, not relevant for new beam
    beam_out.attrs.pop("overlays", None)
    
    # overlay: collect dimension ranges here
    overlay_dims = {}
    

    for dim, params in kwargs.items():
        #print(f"  {dim}: {params}")
        # unpack params
        start, stop, unit = params

        if dim not in beam.dims:
            raise ValueError(f"Dimension '{dim}' not found in the xarray object.")
        
        # check unit compatibility
        if unit == beam.coords[dim].units:
            # print(f"unit match for dimension '{dim}': {unit}")
            pass
        else:
            start = rescale_by_units(start, unit, beam.coords[dim].units)
            stop = rescale_by_units(stop, unit, beam.coords[dim].units)
            # print(f"converted units for dimension '{dim}': {unit} -> {beam.coords[dim].units}")

        # print(f"Cropping dimension '{dim}' from {start} to {stop} {beam.coords[dim].units}")

        # ToDo: handle if range is larger than coordinate axis
        # ToDo: handle reversed ranges

        # crop:
        beam_out = beam_out.sel(
            {dim: slice(start, stop)}
        )

        # calculate statistics:
        if calc_statistics:
            beam_out = set_statistics(beam_out)

        # set label of new beam
        if label is None:
            beam_out.attrs["label"] = f"{beam.attrs['label']}: cropped"

        else:
            beam_out.attrs["label"] = label
        
    
        # overlay: store overlay info per dimension
        overlay_dims[dim] = {
            "start": start,
            "stop": stop,
            "unit": beam.coords[dim].units,
        }

    # append overlay to input beam
    if overlay_add == True:

        if overlay_label is None:
            overlay_label = f"{beam_out.attrs.get('label')}"

        # add overlay meta data to da, use default style
        add_rectangle_overlay(beam, 
            overlay_dims = overlay_dims, 
            label = overlay_label,
        )

    return beam_out


# reduce/project along one dimesion using certain range
def cross_section(
    da,
    method="mean",
    label=None,
    overlay_add=True,
    overlay_label=None,
    calc_statistics = True,
    **kwargs
):
    """
    Make a cross-section of a DataArray along one dimension within a specified range.

    Parameters
    ----------
    method : str, default='mean', options='mean','sum'
        Explanation of method

    Note
    ----------
    define dimension for cross section in **kwars
    x=(-4, 4, 'cm') <-> dim=(start, stop, unit)

    Example
    ----------
        cross_section(da, x=(-4, 4, 'cm'), method="mean")
    """
    
    # --- Step 1: validate kwargs ---
    if len(kwargs) == 0:
        raise ValueError("No dimension specified. Provide one dimension like x=(start, stop, unit).")
    
    if len(kwargs) > 1:
        raise ValueError("Only one dimension can be reduced at a time.")
    
    # extract dimension and params
    dim, params = next(iter(kwargs.items())) # such syntax, wow, 

    # validate params
    if not isinstance(params, (tuple, list)) or len(params) != 3:
        raise ValueError(
            f"Expected (start, stop, unit) for dimension '{dim}', got {params}"
        )
    
    start, stop, unit = params

    # validate dimension exists
    if dim not in da.dims:
        raise ValueError(f"Dimension '{dim}' not found in DataArray.")

    #print(f"Cross section: '{dim}' from {start} to {stop} {unit} using '{method}'")

    # --- Step 2: copy input (same pattern as crop) ---
    da_out = da.copy()
    da_out.attrs = dict(da.attrs)

    # remove existing overlays, not relevant for new beam
    da_out.attrs.pop("overlays", None)

    # --- Step 3: unit handling ---
    coord = da.coords[dim]

    if not hasattr(coord, "units"):
        raise AttributeError(f"Coordinate '{dim}' has no 'units' attribute.")

    coord_unit = coord.units

    if unit == coord_unit:
        # print(f"Unit match for dimension '{dim}': {unit}")
        pass
    else:
        start = rescale_by_units(start, unit, coord_unit)
        stop = rescale_by_units(stop, unit, coord_unit)
        print(f"Converted units for '{dim}': {unit} -> {coord_unit}")

    #print(f"Selecting range {start} to {stop} {coord_unit} along '{dim}'")

    # --- Step 4: slice ---
    da_sel = da_out.sel({dim: slice(start, stop)})

    # --- Step 5: reduction ---
    if method == "mean":
        da_cross_section = da_sel.mean(dim=dim)
    elif method == "sum":
        da_cross_section = da_sel.sum(dim=dim)
    else:
        raise ValueError(f"Unknown method '{method}'. Use 'mean' or 'sum'.")
    
    # calculate statistics:
    if calc_statistics:
        da_cross_section = set_statistics(da_cross_section)

    # set label of new beam
    if label is None:
        da_cross_section.attrs["label"] = f"{beam.attrs['label']}: cross section"

    else:
        da_cross_section.attrs["label"] = label


    # append attributes describing reduction 
    da_cross_section.attrs["cross section"] = {
        "dim": dim,
        "start": start,
        "stop": stop,
        "unit": coord_unit,
        "method": method,
    }
        # --- Step 6: add overlay to original DataArray ---
    if overlay_add:
        # add overlay attribute if not there yet
        if "overlays" not in da.attrs:
            da.attrs["overlays"] = []

        # set dimensions
        overlay_dims = {}

        overlay_dims[dim] = {
            "start": start,
            "stop": stop,
            "unit": coord_unit,
        }
        
        if overlay_label is None:
            overlay_label = f"{da_cross_section.attrs.get('label')}"

        # add overlay meta data to da, use default style
        add_rectangle_overlay(da, 
            overlay_dims = overlay_dims, 
            label = overlay_label,
        )

    return da_cross_section 



# ToDo: type hint for xarray
def set_Energy(beam,
    energy,
    unit_energy,
    #unit_Fluence = "J/cm2",
    ):
    
    # convert Energy to Joule
    energy = rescale_by_units(energy, unit_energy, "J")

    # get x and y coordinate (checkes that they exists)
    coord_x = beam["x"]
    coord_y = beam["y"]

    # get pixel size (also checks equality)
    pixel_size_x = pixel_size(coord_x)
    pixel_size_y = pixel_size(coord_x)

    # convert to 'cm' (checks that coordinate unit needs to be unit of length)
    pixel_size_x = rescale_by_units(pixel_size_x,coord_x.attrs['units'],'cm')
    pixel_size_y = rescale_by_units(pixel_size_y,coord_y.attrs['units'],'cm')

    # pixel area
    pixel_area = pixel_size_x * pixel_size_y

    # normalization
    beam.values  = beam.values * energy / np.sum(beam.values) / pixel_area

    # set data array unit
    beam.attrs['units']='J/cm\u00b2'

    return beam

# ToDo: add other operations
def beam_arithmetic(da: xr.DataArray, operation: str, operand: float = None, label: str = None) -> xr.DataArray:
    """
    Perform arithmetic operation on the data of an xarray DataArray and update the label.
    

    """
    # perform operation
    if operation == 'normalize_peak':
        if operand is None:
            operand = 1 
        data_new = operand * da.values / np.max(da.values)
    else:
        raise ValueError(f"Unsupported operation '{operation}'. Use 'normalize_peak'.")
    
    # new data array
    da_new = da.copy(data=data_new)
    if label is not None:
        da_new.attrs['label'] = label
    
    return da_new

if __name__ == "__main__":
    import laser_beam as lb

    import matplotlib.pyplot as plt

    # define beam
    beam = lb.create_beam_xy(
        type = "Gauss",
        func_params={
            'width_x': 20,
            'width_y': 20,
        },
        axis_unit="mm",
        label='Gauss',
        units = 'W/cm²'
    )

    # process beam
    beam_cropped = crop(beam,
        x=(-10, 10, 'mm'),
        y=(-20, 20, 'mm'),
    )
    
    # make cross section
    beam_lineout = cross_section(beam, 
        y = (-1,1,'mm'),
        label = 'Lineout Gauss',
    )


    # plot
    fig, ax  = plt.subplots(1, 3,figsize=(15,4))
    
    lb.plot_2D(beam,
        ax = ax[0],
        plot_unit_x='cm',
        plot_unit_z='mm²',
    )
    lb.plot_2D(beam_cropped,
        ax = ax[1],
        plot_unit_y='cm',
    )
    lb.plot_1D(beam_lineout,
        ax = ax[2],
        legend_show=False,
    )

    plt.show()


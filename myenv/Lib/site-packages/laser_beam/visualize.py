import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable     # handles positioning of colorbar
import matplotlib.colors as colors

# import all functions from all modules in laser_beam
from laser_beam.utils import rescale_by_units, convert_coord, get_units

__all__ = ['plot_2D', 'plot_1D',
            'add_rectangle_overlay',
            'STYLE_POINTS', 'STYLE_FIT',
            'STYLE_M2_FIT',
            ]

def plot_1D(
    da,
    plot_styles=None,
    title = None,
    title_show = True,
    x_plot_units = None,
    x_label_show = True,
    x_label = None,
    y_plot_units = None,
    y_label_show = True,
    y_label = None,
    legend_show = True,
    legend_kwargs=None,
    plot_units_brackets = ('(',')'),
    overlays_show = True,
    overlays_label_show = True,
    ax=None,
    **kwargs
    ):

    # validate input: either a single DataArray or a list of DataArrays
    if isinstance(da, xr.DataArray):
        da_list = [da]
    elif isinstance(da, list) and all(isinstance(d, xr.DataArray) for d in da):
        da_list = da
    elif isinstance(da, xr.Dataset):
        # take all data variables from dataset
        da_list = list(da.data_vars.values())
        if title is None:
            title =da.attrs.get('title',None)
            print(f"title in plot: {title}")

    else:
        raise ValueError("Input must be an xarray DataArray.")
    
    # Normalize styles to a list of dicts
    if plot_styles is None:
        plot_styles = [{}] * len(da_list)
    elif isinstance(plot_styles, dict):
        plot_styles = [plot_styles] * len(da_list)
    elif isinstance(plot_styles, list):
        # Optional: check if lengths match
        if len(plot_styles) != len(da_list):
            print(f"Warning: styles length ({len(plot_styles)}) != da_list length ({len(da_list)})")
    else:
        raise ValueError("plot_styles must be a dict, a list of dicts, or None.")
    
    # copy to avoid mutating the caller's DataArray
    da_list = [da.copy() for da in da_list]

    # prepare plot styles and merge with existing ones in da.attrs
    da_list = [prepare_da_plot_style(da, style) for da, style in zip(da_list, plot_styles)]

    # determin target unis for plotting
    first_coord = da_list[0].coords[da_list[0].dims[0]]
    first_units = get_units(first_coord)
    first_units_y = da_list[0].units

    x_target_units = x_plot_units if x_plot_units is not None else first_units
    y_target_units = y_plot_units if y_plot_units is not None else first_units_y

    coord_x_names = []
    coord_y_names = []

    # get dimensions, coordinates and units
    for idx, da in enumerate(da_list):
        # x axis
        dim = da.dims[0]
        coord = da.coords[dim]
                
        x_units = get_units(coord)
        y_units = da.units
        coord_x_name = coord.attrs.get('long_name', dim)  # get long name if exists otherwise use dim name
        coord_y_name = da.attrs.get('long_name', da.name)  # get long name if exists otherwise use da name

        #print(f"da: {da.name}, corrd_y_name: {coord_y_name}")
        
        # convert units if necessary
        if x_units != x_target_units:

            new_coord = rescale_by_units(coord.values, x_units, x_target_units)

            da = da.assign_coords({dim: (coord.dims, new_coord)})   
            da.coords[dim].attrs['units'] = x_target_units
        
            # write back to list
            da_list[idx] = da
            x_units = x_target_units
        
        if y_units != y_target_units:

            da = da.copy(data=rescale_by_units(da.values, y_units, y_target_units))
            da.attrs['units'] = y_target_units

            # write back to list
            da_list[idx] = da
            y_units = y_target_units

        # get all coordinate names for labeling
        if coord_x_name not in coord_x_names:
            coord_x_names.append(coord_x_name)


        # # --- y axis --- da.units and da.name
        # # check units (ToDo: conversion if necessary)
        # if da.units != first_units_y:
        #     raise ValueError(f"not using the same units: '{first_units_y}' & '{da.units}'")

        if coord_y_name not in coord_y_names:
            coord_y_names.append(coord_y_name)

     # --- plotting ---
    if ax is None:
        ax = plt.gca()

    for da in da_list:
        # extact coordinate for x axis
        dim = da.dims[0]
        coord = da.coords[dim]

        # get plot style for this da
        plot_style = da.attrs.get("plot_style", {})

        # determin the "kind" of plot (default to line)
        plot_kind = plot_style.pop("kind", "line")

        # Dispatch Matplotlib
        if plot_kind == 'line':
            ax.plot(coord.values, da.values, **plot_style, **kwargs)
        #ToDo: implement this properly with plot styles and kwargs, needed for histogramm
        elif plot_kind == 'bar':
            print(f"Plotting '{da.name}' as bar plot")
            ax.bar(coord.values, da.values) # ToDo: add plot_styles
        else:
            raise ValueError(f"Unsupported plot kind: '{plot_kind}'. Supported kinds are 'line' and 'bar'.")
    
    # plot overlays
    if overlays_show == True:
        for da in da_list:
            dim = da.dims[0]
            apply_overlays(ax, da, 
                dim, da.name, 
                x_target_units, y_target_units, 
                show_labels = overlays_label_show,
                )

    # --- set labels ----
    # title
    if title_show:
        if title is None:
            title = da.attrs.get('label', da.name)
        ax.set_title(title)

    # get brackets for units
    left_br, right_br = plot_units_brackets

    # x axis label
    if x_label_show:
        if x_label is None:
            x_label = f"${', '.join(coord_x_names)}$ {left_br}{x_target_units}{right_br}"
        ax.set_xlabel(x_label)

    # y axis label
    if y_label_show:
        if y_label is None:
            y_label = f"${', '.join(coord_y_names)}$ {left_br}{y_target_units}{right_br}"
        ax.set_ylabel(y_label)

    # set legend
    if legend_show:
        legend_labels = [da.attrs.get('label', da.name) for da in da_list]
        ax.legend(legend_labels,**(legend_kwargs or {}))



    return ax 



def plot_2D(
    da: xr.DataArray,
    log_scale = False,
    value_range = None,            # options: None or (vmin, vmax)
    coord_range = None,            # 
    plot_complex = 'abs',           # opions: 'abs', 'angle', 'real', 'imag'
    title = None,
    title_show = True,
    x_label_show = True,
    y_label_show = True,
    secondary_axis = None,          # options: ('None','angle', 'focus size')
    plot_unit_x = None,
    plot_unit_y = None,
    plot_unit_z = None,
    plot_units_brackets = ('(',')'),
    cbar_show = True,
    cbar_label_show = True,
    cmap="jet", # 'viridis'
    overlays_show = True,
    overlays_label_show = True,
    aspect_match = True,  # handled by units
    ax=None,
    **kwargs):
    
    """
    Docstring for plot_2D
    
    :param da: Description
    :type da: xr.DataArray
    :param log_scale: Description
    :param value_range: Description
    :param coord_range: Description
    :param plot_complex: Description
    :param title: Description
    :param title_show: Description
    :param x_label_show: Description
    :param y_label_show: Description
    :param secondary_axis: Description
    :param plot_unit_x: Description
    :param plot_unit_y: Description
    :param plot_unit_z: Description
    :param plot_units_brackets: Description
    :param cbar_show: Description
    :param cbar_label_show: Description
    :param overlays_show: Description
    :param cmap: Description
    :param aspect: Description
    :param ax: Description
    :param kwargs: Description
    """ 

    # Copy to avoid mutating the caller's DataArray
    da = da.copy()


    # Get dims, coords and units
    # ToDo: handle higher dimensions, maybe first 2 non-singleton dimensions

    x_dim, y_dim = da.dims[:2]
    coord_x_name = x_dim
    coord_y_name = y_dim

    coord_names = list(da.coords.keys())

    # handle secondary axes
    if secondary_axis == 'angle':
        # check if angle_x and y secondary coordinates excists
        if 'θ_x' in da.coords and 'θ_y' in da.coords:
            coord_x_name = 'θ_x'
            coord_y_name = 'θ_y'
            #print(f"coordinates 'θ_x' and 'θ_y' are part of coordinates: {coord_names}")
        else:
            print(f"coordinates 'θ_x' and 'θ_y' are not part of coordinates: {coord_names}")
    elif secondary_axis == 'focus size':
        if'x_focus' in da.coords and 'y_focus' in da.coords:
            coord_x_name = 'x_focus'
            coord_y_name = 'y_focus'
            #print(f"coordinates 'x_focus' and 'y_focus' are part of coordinates: {coord_names}")
    else:
        pass
        #print(f"Using primary coordinates: {coord_x_name} and {coord_y_name}")

    # Attempt unit conversion if requested
    da = convert_coord(da, coord_x_name, plot_unit_x)
    da = convert_coord(da, coord_y_name, plot_unit_y)
    # da = convert_coord(da, coord_z_name, plot_unit_z)

    coord_x = da.coords[coord_x_name]
    coord_y = da.coords[coord_y_name]

    # get 
    unit_x = get_units(coord_x)
    unit_y = get_units(coord_y)

    # print(f"x: 'plot unit' {plot_unit_x} -> 'unit' {unit_x}")
    # print(f"y: 'plot unit' {plot_unit_y} -> 'unit' {unit_y}")
    



    # coordinate label
    if da.attrs.get('long_name') is not None:
        cbar_name = da.attrs['long_name']
    else:
        cbar_name = da.name
    if coord_x.attrs.get('long_name') is not None:
        coord_x_label = coord_x.attrs['long_name']
    else:
        coord_x_label = coord_x_name

    if coord_y.attrs.get('long_name') is not None:
        coord_y_label = coord_y.attrs['long_name']
    else:
        coord_y_label = coord_y_name



    # handle range of coordinates
    if coord_range is not None and np.isscalar(coord_range):
        # symmetric range around 0
        coord_range = [-coord_range, coord_range,-coord_range, coord_range]


    if plot_complex == 'abs':
        da = np.abs(da)
    elif plot_complex == 'angle':
        #da = np.angle(da)
        da = da.copy(data=np.angle(da))
    elif plot_complex == 'real':
        da = np.real(da)
    elif plot_complex == 'imag':
        da = np.imag(da)
    else:
        raise ValueError(f"Invalid plot_complex option: '{plot_complex}'. "
                            f"Expected 'abs', 'angle', 'real', or 'imag'.")

    # lin vs log scale

    if log_scale == True:

        # Log scale requires strictly positive values, set's negative values to NaN
        da = da.where(da > 0)
        
        if value_range is None:
            kwargs["norm"] = colors.LogNorm()
        else: # set vmin and vmax for log scale
            kwargs["norm"] = colors.LogNorm(vmin=value_range[0], vmax=value_range[1])


    if ax is None:
        # make new figure
        fig, ax = plt.subplots()

    # print(da.dims)
    # for d in da.dims:
    #     print(d, da.coords[d].values)

    # note that da.plot() handles centering of pixels extends range by pixel_size/2 to each side
    # common cryptic error if ax[0] is passed instead of ax[0,0] with subplots
    plot_obj = da.plot(
        x = coord_x_name,
        y = coord_y_name,
        cmap = cmap,
        add_colorbar=False, # add colorbar manually (correct position)
        ax = ax,
        **kwargs
    )

    # handle coordinate range
    if coord_range is not None:
        ax.set_xlim(coord_range[0], coord_range[1])
        ax.set_ylim(coord_range[2], coord_range[3])

    # handle aspect ratio when not defined (this really wasn't fun)
    if aspect_match:
        if unit_x == unit_y:
            ax.set_aspect('equal')
            print("units x/y are same -> aspect = 1")
        else:
            # if units are compatible, set aspect using unit conversion
            try:
                # get data range
                x_range = abs(ax.get_xlim()[1] - ax.get_xlim()[0])
                y_range = abs(ax.get_ylim()[1] - ax.get_ylim()[0])
                # get ratio of units
                aspect = rescale_by_units(1, unit_y, unit_x)
                # rescale box instead of setting aspect (doesn't work if = 10, aspect OK, but size off)
                ax.set_box_aspect(y_range / x_range * aspect)
            
            except ValueError as e:
                print(f"With aspect_match: {e}")


    # colorbar positioning magic
    if cbar_show:
        cax = ax.inset_axes([1.05, 0, 0.05, 1])  # [x, y, width, height] in axes coords
        cbar = plt.colorbar(plot_obj, cax=cax)

    # get brackets for units
    left_br, right_br = plot_units_brackets

    # set color bar label
    if cbar_show == True and cbar_label_show == True:
        cbar.set_label(f"${cbar_name}$ {left_br}{get_units(da)}{right_br}")

    # set x and y label
    if x_label_show == True:    
        xlabel = f"${coord_x_label}$ {left_br}{unit_x}{right_br}"
        ax.set_xlabel(xlabel)
    else:
        ax.set_xlabel("")
    if y_label_show == True:
        ylabel = f"${coord_y_label}$ {left_br}{unit_y}{right_br}"
        ax.set_ylabel(ylabel)
    else:
        ax.set_ylabel("")


    # set title
    if title_show:
        if title is None:
            title = da.attrs.get('label', da.name)
        ax.set_title(title)

    # plot overlays:
    if overlays_show == True:
        apply_overlays(ax, da, 
            x_dim, y_dim, 
            unit_x, unit_y, 
            show_labels = overlays_label_show,
        )
        

    return ax, plot_obj

# some helper functions
# apend rectangle over lay
def add_rectangle_overlay(beam, overlay_dims, label, rect_style=None, label_style=None):
    if "overlays" not in beam.attrs:
        beam.attrs["overlays"] = []

    rect_style_default = {
        "edgecolor": "r",
        "facecolor": (1, 0, 0, 0.4),
        "linewidth": 1,
    }

    label_style_default = {
        "fontsize": 10,
        "color": "red",
        "ha": "left",
        "va": "bottom",
        "bbox": {
            "facecolor": "white",
            "alpha": 0.5,
            "edgecolor": "none",
            "pad": 1,
        },
    }

    beam.attrs["overlays"].append({
        "type": "rectangle",
        "dims": overlay_dims,
        "label": label,
        "rect_style": {**rect_style_default, **(rect_style or {})},
        "label_style": {**label_style_default, **(label_style or {})},
    })

# (obselete, subset of rectangle) helper function to draw 
def draw_box(ax, overlay, x_params, y_params, show_label):
    """Logic specifically for rendering a box and its label."""
    dims = overlay.get("dims", {})
    style = overlay.get("style", {})
    
    # --- X Bounds ---
    if x_params['dim_name'] in dims:
        d = dims[x_params['dim_name']]
        x0, x1 = rescale_by_units([d["start"], d["stop"]], d['unit'], x_params['unit'])
    else:
        x0, x1 = ax.get_xlim()

    # --- Y Bounds ---
    if y_params['dim_name'] in dims:
        d = dims[y_params['dim_name']]
        y0, y1 = rescale_by_units([d["start"], d["stop"]], d['unit'], y_params['unit'])
    else:
        y0, y1 = ax.get_ylim()

    # --- Geometry Calculation ---
    x_plot, y_plot = min(x0, x1), min(y0, y1)
    width, height = abs(x1 - x0), abs(y1 - y0)

    # Add the Rectangle
    rect = Rectangle((x_plot, y_plot), width, height, **style)
    ax.add_patch(rect)

    # --- Label Logic ---
    # ToDo: add label style
    label = overlay.get("label")
    if show_label and label:
        # Calculate offset (2% of current view)
        dx = 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0])
        dy = 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
        
        ax.text(
            x_plot + dx, y_plot + dy, label,
            fontsize=10, color='red',
            horizontalalignment='left', verticalalignment='bottom',
            bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1)
        )

# draw rectangle
def draw_rect(ax, overlay, x_params, y_params, show_label):
    """
    Unified renderer for:
      - boxes (both x & y dims present)
      - axvspan-like (only x dim present)
      - axhspan-like (only y dim present)
      - full-axes fallback (no dims)
    """

    dims = overlay.get("dims", {})
    rect_style = overlay.get('rect_style', {})
    label_style = overlay.get('label_style',{})

    # -------------------------
    # X bounds
    # -------------------------
    if x_params["dim_name"] in dims:
        d = dims[x_params["dim_name"]]
        x0, x1 = rescale_by_units(
            [d["start"], d["stop"]],
            d["unit"],
            x_params["unit"]
        )
    else:
        x0, x1 = ax.get_xlim()

    # -------------------------
    # Y bounds
    # -------------------------
    if y_params["dim_name"] in dims:
        d = dims[y_params["dim_name"]]
        y0, y1 = rescale_by_units(
            [d["start"], d["stop"]],
            d["unit"],
            y_params["unit"]
        )
    else:
        y0, y1 = ax.get_ylim()

    # -------------------------
    # Geometry
    # -------------------------
    x_plot, y_plot = min(x0, x1), min(y0, y1)
    width = abs(x1 - x0)
    height = abs(y1 - y0)

    rect = Rectangle((x_plot, y_plot), width, height, **rect_style)
    ax.add_patch(rect)

    # -------------------------
    # Label
    # -------------------------
    label = overlay.get("label")
    if show_label and label:
        dx = 0.02 * (ax.get_xlim()[1] - ax.get_xlim()[0])
        dy = 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])

        # split bbox from text kwargs
        text_kwargs = {k: v for k, v in label_style.items() if k != "bbox"}
        bbox_kwargs = label_style.get("bbox", None)

        ax.text(
            x_plot + dx,
            y_plot + dy,
            label,
            bbox=bbox_kwargs,
            **text_kwargs,
        )

# helper function to apply helper functions
def apply_overlays(ax, da, x_dim, y_dim, unit_x, unit_y, show_labels):
    """
    Dispatcher to handle different overlay types.
    Can be called from plot_1D or plot_2D.
    """
    # pack dimension and unit for the x and y axes
    x_params = {'dim_name': x_dim, 'unit': unit_x}
    y_params = {'dim_name': y_dim, 'unit': unit_y}

    # Map overlay types to their respective drawing functions
    dispatch_map = {
        "box": draw_box,
        'rectangle': draw_rect,
        # "line": draw_line,  # Future expansion
    }

    # loop through overlays:
    for overlay in da.attrs.get("overlays", []):
        overlay_type = overlay.get("type")
        
        if overlay_type in dispatch_map:
            # Call the specific function (e.g., draw_box)
            dispatch_map[overlay_type](ax, overlay, x_params, y_params, show_label=show_labels)
        else:
            print(f"Warning: overlay type '{overlay_type}' not defined.")

def prepare_da_plot_style(da, plot_style=None):
    # Copy to avoid mutating the caller's DataArray
    da_ready = da.copy()

    # merge plot style with existing one in da.attrs
    style_base = da_ready.attrs.get("plot_style", {})
    if plot_style:
        style_base.update(plot_style)

    # write back to da_ready
    da_ready.attrs["plot_style"] = style_base

    return da_ready


STYLE_POINTS = {
    'kind': 'line',
    'linestyle': 'None',
    'marker': 'o',
    'markerfacecolor': 'none',   # makes it hollow
}

STYLE_FIT = {
    'kind': 'line',
    'linestyle': '-',
    'marker': 'None',
    'linewidth': 1,  # Thin line
    'alpha': 0.8,    # Slightly transparent so data pops
}

# colors I like
color1 = '#ec0868'
color2 = '#fc2f00'
color3 = '#ec7d10'
color4 = '#ffbc0a'


# some style template
STYLE_M2_FIT = [STYLE_POINTS| {"color": color2},
                STYLE_POINTS| {"color": color4},
                STYLE_FIT| {"color": color2},
                STYLE_FIT| {"color": color4},
                ]
# some testing
if __name__ == "__main__":
     
    print("running visualize.py as main")
    print("break here")

    import laser_beam as lb

    beam_1 = lb.create_beam_xy(
        label = 'Beam 1',
        name='Height',
        units='miles',
        func_params={
            'width_x': 10,
            'width_y': 15,
            'amplitude': 1,
        },
        axis_unit='mm',
        axis_pixelsize=0.1,
        axis_x_N=200,
        axis_y_N=200,
    )

    beam_2 = lb.create_beam_xy(
        label = 'Beam 2',
        name='Altitude',
        units='m',
        func_params={
            'width_x': 1,
            'width_y': 1.5,
            'amplitude': 1600,
        },
        axis_unit='cm',
        axis_pixelsize=0.01,
        axis_x_N=200,
        axis_y_N=200,
    )
    
    lineout_1 = lb.cross_section(beam_1,
        y=(-1,1,'mm'),
        label = 'Lineout 1',
        overlay_label='',
    )

    lineout_2 = lb.cross_section(beam_2,
        x=(-0.2,+0.1,'cm'),
        label = 'Lineout 2',
        overlay_label='LO 2',
    )

    
    
    fig,ax = plt.subplots(1, 3,figsize=(15,4))
    
    plot_2D(beam_1,ax=ax[0])
    plot_2D(beam_2,
        plot_unit_x='inch',
        overlays_label_show=False,
        ax=ax[1],
        )
    plot_1D([lineout_1, lineout_2],
        y_plot_units='cm',
        ax=ax[2],
        )
    #plot_1D(lineout_1, ax=ax[2])

    plt.tight_layout()
    plt.show()


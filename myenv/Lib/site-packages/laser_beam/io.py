"""Docstring for laser_beam.io"""

import xarray as xr
import pandas as pd

import numpy as np
from pathlib import Path

from PIL import Image, PngImagePlugin
import json

import os
import io

__all__ = ['save_dataarray_as_png', 'load_image_as_dataarray',
           'flat_dict_to_dataset', 'dataset_to_flat_dict',
           'load_table_to_dataset', 'flat_dict_to_excel',  'flat_dict_to_excel_bytes',
           'load_table_to_flat_dict',]

# save laser beam data array to png
def save_dataarray_as_png(
    data: xr.DataArray,
    file_name: str,
    folder: str = None
) -> None:
    """
    Save a 2D xarray DataArray as a PNG file (grayscale) with metadata.
    
    Parameters:
        data (xr.DataArray): The 2D DataArray to save.
        file_name (str): Name of the PNG file to save.
        folder (str, optional): Folder to save the file in.
    """
    print("running: save_dataarray_as_png()")

    # 1. Setup file path
    if folder is None:
        file_path = file_name
    else:
        os.makedirs(folder, exist_ok=True)
        file_path = os.path.join(folder, file_name)
    
    
    # 2. Validate input
    if data.ndim != 2:
        raise ValueError(f"DataArray must be 2D, got {data.ndim} dimensions")
    
    dim_names = list(data.dims)
    
    # 3. Prepare image data
    # Extract numpy array
    array = data.values
    
    # Store original min/max for later reconstruction
    data_min = float(np.min(array))
    data_max = float(np.max(array))
    
    # Flip upside down (image display convention)
    array = np.flipud(array)
    
    # Shift min to 0 (avoid negative values)
    array_norm = array - data_min
    
    # Normalize to 0-1
    if data_max != data_min:
        array_norm = array_norm / (data_max - data_min)
    else:
        array_norm = np.zeros_like(array_norm)  # Handle constant array case
    
    # Convert to 16-bit unsigned integer (2^16 - 1 = 65535)
    array_uint16 = (array_norm * 65535).astype(np.uint16)
    
    # 4. Create PIL image
    image = Image.fromarray(array_uint16, mode='I;16')
    
    # 5. Prepare metadata
    meta = PngImagePlugin.PngInfo()
    
    # Add format indicator
    meta.add_text("type", "xarray_dataarray")
    
    # 6. Serialize DataArray metadata
    
    # Save dimension names
    meta.add_text("dims", json.dumps(dim_names))
    
    # Save coordinates (arrays + attributes)
    coords_data = {}
    for coord_name in data.coords:
        coords_data[coord_name] = {
            'values': data.coords[coord_name].values.tolist(),
            'attrs': dict(data.coords[coord_name].attrs)
        }
    meta.add_text("coords", json.dumps(coords_data))

    # Save main DataArray attributes
    meta.add_text("attrs", json.dumps(dict(data.attrs)))

    # Save DataArray name
    meta.add_text("name", str(data.name) if data.name is not None else "")
    
    # Save min/max for reconstruction
    meta.add_text("data_min", str(data_min))
    meta.add_text("data_max", str(data_max))

    
    # 7. Save image with metadata
    image.save(file_path, pnginfo=meta)

# load png
def load_image_as_dataarray(
    file_name: str,
    folder: str = None,
    # Fallback parameters for cases without metadata:
    default_name: str = "intensity",
    default_unit: str = "a.u.",
    default_axis_unit: str = "px",
    default_pixel_size: float = 1.0
) -> xr.DataArray:
    """
    Load a PNG, JPG, TIFF, BMP, etc. file as a 2D xarray DataArray.
    
    Handles three cases:
    1. Annotated PNG with xarray metadata (type="xarray_dataarray")
    2. ELI Camera PNG with MaxValue metadata
    3. Generic raw Image (PNG, JPG, TIFF, BMP, etc.) without metadata
    
    Parameters:
        file_name (str): Name of the PNG file to load.
        folder (str, optional): Folder path where the file is located.
        default_name (str): Default name for DataArray if not in metadata.
        default_unit (str): Default unit for data values if not in metadata.
        default_axis_unit (str): Default unit for axes if not in metadata.
        default_pixel_size (float): Default pixel size if not in metadata.
    
    Returns:
        xr.DataArray: 2D DataArray with reconstructed metadata.
    """
    
    print("running: load_image_as_dataarray()")
    import debugpy
    print(f"Is debugger attached? {debugpy.is_client_connected()}")
    breakpoint()

    # 1. Setup file path
    if folder is None:
        file_path = file_name
    else:
        file_path = os.path.join(folder, file_name)
    
    # 2. Load image with PIL
    image = Image.open(file_path)
    
    # Extract metadata (will be empty dict {} for formats without metadata support)
    metadata = image.info
    
    # 3. Detect PNG type
    image_type = metadata.get("type")
    
    if image_type == "xarray_dataarray":
        load_case = "annotated"
    elif "MaxValue" in metadata:
        load_case = "eli_camera"
    else:
        load_case = "raw"

    print(f"load_case: {load_case}")
    
    # 4. Load data array
    data = np.array(image, dtype=np.float64)
    
    # Validate that image is 2D grayscale
    if data.ndim != 2:
        mode = image.mode
        raise ValueError(
            f"Loaded image must be grayscale (2D array).\n"
            f"Got {data.ndim} dimensions with shape {data.shape}.\n"
            f"Image mode is: {mode}"
        )
    
    # 5. Case 1: Annotated xarray PNG
    if load_case == "annotated":
        
        # Deserialize metadata
        dims = json.loads(metadata.get("dims"))
        coords_data = json.loads(metadata.get("coords"))
        attrs = json.loads(metadata.get("attrs"))
        name = metadata.get("name")
        if name == "":
            name = None
        data_min = float(metadata.get("data_min"))
        data_max = float(metadata.get("data_max"))
        
        # Flip upside down (reverse save operation)
        data = np.flipud(data)
        
        # Reconstruct intensity from 0-65535 range to original range
        data = data / 65535.0  # normalize to 0-1
        data = data * (data_max - data_min) + data_min  # scale to original range
        
        # Reconstruct coordinates
        coords = {}
        for coord_name, coord_info in coords_data.items():
            coords[coord_name] = (
                coord_name,
                np.array(coord_info['values']),
                coord_info['attrs']
            )
        
        # Create DataArray
        dataarray = xr.DataArray(
            data,
            dims=dims,
            coords=coords,
            name=name,
            attrs=attrs
        )
        
        return dataarray
    
    # 6. Case 2: ELI Camera PNG
    elif load_case == "eli_camera":
        # Extract MaxValue from metadata
        max_value = float(metadata.get("MaxValue"))
        
        # Flip upside down (convention)
        data = np.flipud(data)
        
        # Rescale from 0-65535 range to 0-MaxValue
        data = (data / 65535.0) * max_value
        
        # Get shape to build coordinates
        Nx, Ny = data.shape
        
        # Build coordinate arrays (centered at 0)
        x = default_pixel_size * (np.arange(Nx) - (Nx - 1) / 2)
        y = default_pixel_size * (np.arange(Ny) - (Ny - 1) / 2)
        
        # Create DataArray with basic metadata
        dataarray = xr.DataArray(
            data,
            dims=['x', 'y'],
            coords={
                'x': ('x', x, {'units': default_axis_unit}),
                'y': ('y', y, {'units': default_axis_unit}),
            },
            name=default_name,
            attrs={'units': default_unit}
        )
        
        return dataarray
    
    
    # 7. Case 3: Generic raw image (any grayscale format)
    else:
        # Flip upside down (convention)
        data = np.flipud(data)
        
        # Get shape to build coordinates
        Nx, Ny = data.shape
        
        # Build coordinate arrays (centered at 0)
        x = default_pixel_size * (np.arange(Nx) - (Nx - 1) / 2)
        y = default_pixel_size * (np.arange(Ny) - (Ny - 1) / 2)
        
        # Create DataArray with default metadata
        dataarray = xr.DataArray(
            data,
            dims=['x', 'y'],
            coords={
                'x': ('x', x, {'units': default_axis_unit}),
                'y': ('y', y, {'units': default_axis_unit}),
            },
            name=default_name,
            attrs={'units': default_unit}
        )
        
        return dataarray
    
    # 8. Return DataArray
    # TODO: return the created DataArray

LOAD_TABLE_CONFIG_DEFAULT = {
    'header': None,             # ['keys']if None, load all key-value pairs until first empty row; otherwise, only load specified keys
    'anchor': 'Coordinate',
    'columns': None,          # ['keys'] specify which columns to load besides anchor (if None, load all)
}

# load excel or .txt file to dict
def load_table_to_flat_dict(
    file_name: str,
    folder: str = None,
    config: dict = LOAD_TABLE_CONFIG_DEFAULT,
    #mode: str = 'strict',
    ) -> dict:

    # setup file path
    if folder is None:
        file_path = Path(file_name)
    else:
        file_path = Path(folder) / file_name


    # load data
    if file_path.suffix.lower() in ['.xlsx', '.xls']:
            raw = pd.read_excel(file_path, header=None, engine='openpyxl')
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    # parse key-value header rows until first empty row
    header = {}
    found_keys = set()

    for i in range(len(raw)):
        key = str(raw.iloc[i, 0]).strip()
        val = raw.iloc[i, 1]

        # stop if key is empty or NaN
        if not key or key.lower() == 'nan':
            break

        # convert val: NaN -> None
        if pd.isna(val):
            val = None
            print(f"Warning: Value for key '{key}' is NaN, converting to None.")
        else: 
            val = str(val).strip()

        if config['header'] is None or key in config['header']:
            header[key] = val
            found_keys.add(key)

    # Check for missing keys in the config
    if config['header'] is not None:
        missing = [k for k in config['header'] if k not in found_keys]
        if missing:
            raise ValueError(f"Header key(s) not found in data: {missing}")

    # find anchor row
    anchor = config.get('anchor', 'Coordinate') # <- why not config['anchor']? because we want to allow missing anchor with default fallback
    labels_row = None
    for i in range(len(raw)):
        if str(raw.iloc[i, 0]).strip().lower() == anchor.lower():
            labels_row = i
            break
    
    if labels_row is None:
        raise ValueError(f"Could not find anchor '{anchor}' in file.")

    # parse column header (Lables, Dims, Units)
    labels = raw.iloc[labels_row].tolist()   # e.g. ["Coordinate", "X", "Y"]
    dims     = raw.iloc[labels_row + 1].tolist()   # e.g. ["Position", "Length", "Length"]
    units     = raw.iloc[labels_row + 2].tolist()   # e.g. ["mm", "μm", "μm"]

    # data starts after labels (+ optional dims/units rows)
    data_start = labels_row + 3

    # parse data rows
    data_rows = raw.iloc[data_start:].reset_index(drop=True)
    data_rows.columns = labels

    # Drop fully-empty columns
    data_rows = data_rows.dropna(axis=1, how='all')

    # build measurements dict
    col_filter = config.get('columns', None)

    # Build measurements dict (one for each column)
    measurements = {}
    found_cols = set()

    for i, label in enumerate(labels):
        label = str(label).strip()
        if not label or label.lower() == 'nan':
            continue
        if label.lower() == anchor.lower():
            pass  # always include anchor
        # if col_filter is None, include all columns; otherwise, include only specified columns
        elif col_filter is not None and label not in col_filter:
            continue
        else:
            found_cols.add(label)
        dim  = str(dims[i]).strip()  if dims  and i < len(dims)  else ''
        unit = str(units[i]).strip() if units and i < len(units) else ''
        values = data_rows[label].dropna().tolist()
        measurements[label] = {
            'dim':    dim,
            'unit':   unit,
            'values': values,
        }

    # check for missing columns if col_filter is specified
    if col_filter is not None:
        missing_cols = [c for c in col_filter if c not in found_cols]
        if missing_cols:
            raise ValueError(f"Column(s) {missing_cols} specified in config['columns'] not found in {labels}")

    # prepare result dict
    flat_dict = {
        **header,
        'measurements': measurements,
    }
    
    return flat_dict

# convert flat dict to excel file or BytesIO (reverse of load_table_to_flat_dict)
def flat_dict_to_excel(
    flat_dict: dict,
    file_name,          # str, Path, or BytesIO
    folder: str = None,
    coord_name: str = None,
) -> None:

    # only resolve path if it's a string/Path
    if isinstance(file_name, (str, Path)):
        if folder:
            file_name = Path(folder) / file_name
        else:
            file_name = Path(file_name)
    # BytesIO passes straight through to pandas

    measurements = flat_dict['measurements']

    # resolve coordinate column
    if coord_name is None:
        coord_name = next(iter(measurements))
    elif coord_name not in measurements:
        raise ValueError(f"coord_name '{coord_name}' not found in measurements.")

    # reorder so coord is first
    ordered = {coord_name: measurements[coord_name]}
    ordered.update({k: v for k, v in measurements.items() if k != coord_name})

    # build header rows (key-value attrs) 
    attrs = {k: v for k, v in flat_dict.items() if k != 'measurements'}
    header_rows = [[k, v] + [''] * (len(ordered) - 2) for k, v in attrs.items()] # same number of elements than number of columns

    # empty row as separator
    empty_row = [''] * len(ordered)

    # build column header rows (labels, dims, units)
    labels_row = list(ordered.keys())
    dims_row   = [m['dim']  for m in ordered.values()]
    units_row  = [m['unit'] for m in ordered.values()]

    # build data rows
    data_rows = list(zip(*[m['values'] for m in ordered.values()]))

    # assemble all rows
    all_rows = (
        header_rows
        + [empty_row]
        + [labels_row, dims_row, units_row]
        + list(data_rows)
    )

    df = pd.DataFrame(all_rows)
    df.to_excel(file_name, index=False, header=False)

# helper function to get excel bytes (for testing or in-memory use)
def flat_dict_to_excel_bytes(flat_dict: dict, coord_name: str = None) -> bytes:
    buffer = io.BytesIO()
    flat_dict_to_excel(flat_dict, file_name=buffer, coord_name=coord_name)
    return buffer.getvalue()
    

# convert flat dict to dataset
def flat_dict_to_dataset(
    flat_dict: dict,
    coord_name: str = 'Coordinate',  # specify the coordinate column
    sort: str = None    # if specified, sort the dataset by the given coordinate name (e.g. 'time' or 'frequency')
    ) -> xr.Dataset:
    
    # extract the measurements dict (work on a copy to avoid mutating original)
    measurements = flat_dict['measurements'].copy()

    # 2. Extract and setup the shared Coordinate
    coord_info = measurements.pop(coord_name)
    coord_dim = coord_info['dim']  # e.g., 'time' or 'frequency'

    # 3. Create the Dataset with attributes (excluding 'measurements')
    ds = xr.Dataset(attrs={
        k: v for k, v in flat_dict.items()
        if k != "measurements" and v is not None
    })

    # 4. Add the Coordinate to the Dataset
    ds.coords[coord_dim] = (coord_dim, coord_info['values'], {'units': coord_info['unit']})

    # 4. Create the Dataset
    for label, m in measurements.items():
            #
            # add ink coord_dim ('Position')
            ds[label] = (coord_dim, m['values'])
            
            # Add the specific attributes you need for plotting/legend
            ds[label].attrs = {
                'units': m['unit'],
                'label': label,        # 'X' or 'Y'
                'long_name': m['dim']  # 'length'
            }

    if sort:
        if sort not in ds.coords:
            raise ValueError(f"Cannot sort by '{sort}' because it is not a coordinate in the dataset.")
        ds = ds.sortby(coord_dim)

    return ds

# convert dataset to flat dict (reverse of flat_dict_to_dataset)
def dataset_to_flat_dict(
    ds: xr.Dataset,
    coord_key: str  = None,
    ) -> dict:
    
    # default coordinate key is 'Coordinate' if not specified
    if coord_key is None:
        coord_key = 'Coordinate'  # default key for coordinate column in flat dict

    # start with top-level attributes (everything except 'measurements')
    flat_dict = {k: v for k, v in ds.attrs.items()}
    
    measurements = {}
    
    # coordinate first
    for coord_name, coord in ds.coords.items():
        measurements[coord_key] = {
            'dim': coord_name,
            'values': coord.values,
            'unit': coord.attrs.get('units', None)
        }
    
    # data variables
    for label, da in ds.data_vars.items():
        measurements[label] = {
            'dim': da.attrs.get('long_name', label),
            'values': da.values,
            'unit': da.attrs.get('units', None)
        }
    
    flat_dict['measurements'] = measurements
    
    return flat_dict

def load_table_to_dataset(
    file_name: str,
    folder: str = None,
    sort: bool = True,
    **kwargs
    ) -> xr.Dataset:

    result = load_table_to_flat_dict(file_name = file_name, folder=folder, **kwargs)

    ds = flat_dict_to_dataset(result)

    if sort:
        coord_name = list(ds.coords)[0]
        ds = ds.sortby(coord_name)

    return ds


if __name__ == "__main__":

         
    print("running io.py as main")
    print("break here")

    import laser_beam as lb
    import matplotlib.pyplot as plt

    print(f"name: {__name__}")

    folder = r'tests\test_io'
    file = 'Example_V2.xlsx'

    my_config = LOAD_Table_CONFIG_DEFAULT.copy()
    my_config['header'] = ['title']
    #my_config['columns'] = ['y']

    result = load_table_to_dict(folder = folder, file_name=file, config=my_config)

    print(result)

    ds = dict_to_dataset(result)

    #lb.plot_1D([ds['X'],ds['Y']])
    lb.plot_1D(ds)

    print(ds)

    plt.tight_layout()
    plt.show()




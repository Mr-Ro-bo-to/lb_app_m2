# beam propagation functions: 
# main functions: beam aftre propagating certain distance, beam at focus
# helper functions: fourier transform and coordinate handling m->1/m->kx, urad and back

import numpy as np
import xarray as xr

from laser_beam.utils import pixel_size, rescale_by_units

__all__ = ['fft_xy','propagate',
           'Int_to_Efield','Efield_to_Int',
           'append_secondary_rec_coordinates']

# --- main functions --- 

# calculate fourier transform
def fft_xy(da: xr.DataArray,
    direction: str = 'forward',
    label: str = None,
    ) -> xr.DataArray:
    """
    2D fourier transforms the first 2 dimensions of da

    forward (space -> reciprocal space):
    - fft: (x,y)[units] -> (k_x,k_y)[1/units]
    - assumes dims and coord to be ['x','y'] with units to be 'units' ex. 'm', 'mm', or 'px'
    inverse (reciprocal space -> space):
    - ifft: (k_x,k_y)[rad/units] -> (x,y)[units]
    - assumes dims and coord to be ['k_x','k_y'] with units to be 'rad/units' ex. 'rad/m', 'rad/mm', 'rad/px'
    
    
    Parameters
    ----------
    da : xr.DataArray
        Explanation of da.

    direction : str, default='forward', options={'forward', 'inverse'}
        'forward': space -> reciprocal space
        'inverse': reciprocal space ->  space

    Returns
    -------
    output : xr.DataArray
        DataArray with fourier transformed data, dimensions, coordinates, and units
    """

    # calculate fourier transform of data for reciprocal and spatial space (watch the magic order of ifftshift, fft2/ifft2, and fftshift to conserve sum)
    if direction == 'forward':
        dim_1 = 'x'
        dim_2 = 'y'
        # # FFT ishift / fft2 (sum conserved) / fftshfit
        data_new = np.fft.ifftshift(da.values)
        data_new = np.fft.fft2(data_new, norm='ortho')
        data_new = np.fft.fftshift(data_new)
    elif direction == 'inverse':
        dim_1 = 'k_x'
        dim_2 = 'k_x'
        # # iFFT ishift / ifft2 (sum conserved) / fftshfit
        data_new = np.fft.ifftshift(da.values)
        data_new = np.fft.ifft2(data_new, norm='ortho')
        data_new = np.fft.fftshift(data_new)
    else:
        raise ValueError(f"Invalid direction '{direction}', options=['forward','inversere']")

    # handle coordinates
    coord_1 = da.coords[dim_1]
    coord_2 = da.coords[dim_2]

    # size and spacing of coordinate
    n_1= np.size(coord_1.data)
    n_2= np.size(coord_2.data)
    dx_1 = pixel_size(coord_1.data)
    dx_2 = pixel_size(coord_2.data)

    # calculate coordinates for reciprocal and spatial space
    if direction == 'forward':
        # dimension 'x' -> 'k_x'
        dim_1_new = 'k_x'
        dim_2_new = 'k_y'
        # units: 'mm' -> 'rad/mm'
        unit_1_new = f'rad/{coord_1.units}'
        unit_2_new = f'rad/{coord_2.units}'
        # calculate reciprocal coordinate data
        coord_1_data_new = 2 * np.pi * np.fft.fftfreq(n_1, d=dx_1)
        coord_2_data_new = 2 * np.pi * np.fft.fftfreq(n_2, d=dx_2)
        coord_1_data_new = np.fft.fftshift(coord_1_data_new)
        coord_2_data_new = np.fft.fftshift(coord_2_data_new)
    elif direction == 'inverse':
        # dimension  'k_x' -> 'x'
        dim_1_new = 'x'
        dim_2_new = 'y'
        # units:  'rad/mm' -> 'mm' (strip 'rad/')
        unit_1_new = coord_1.units[4:].strip()
        unit_2_new = coord_2.units[4:].strip()
        # calculate spatial coordinate data
        coord_1_data_new = 2 * np.pi * np.fft.fftfreq(n_1, d=dx_1)
        coord_2_data_new = 2 * np.pi * np.fft.fftfreq(n_2, d=dx_2)
        coord_1_data_new = np.fft.fftshift(coord_1_data_new)
        coord_2_data_new = np.fft.fftshift(coord_2_data_new)

    else:
        raise ValueError(f"How did I even get here")

    # assigne name to 
    if label is None:
        label = da.attrs.get('label', da.name)  
    
    # handle beam attributes: copy and update label
    attrs_copy = da.attrs.copy()
    attrs_copy['label'] = label

    # create new DataArray
    da_2 = xr.DataArray(
        data_new,
        dims=[dim_1_new,dim_2_new],
        coords={
            dim_1_new: (dim_1_new, coord_1_data_new, {'units': unit_1_new}),
            dim_2_new: (dim_2_new, coord_2_data_new, {'units': unit_2_new}),
        },
        name=da.name,
        attrs = attrs_copy,
    )

    return da_2

# calculate phase shift for propagation over distance (Fresnel approximation)
def propagate(da: xr.DataArray, wavelength: float, distance: float, label = None, paraxial_aprox = False) -> xr.DataArray:
    """
    Short summary line.

    Longer description if needed.

    Parameters
    ----------
    da : xr.DataArray
        DataArray with dimensions and coordinates ('k_x','k_y') and units 'rad/m' (or convertible to 'rad/m')

    wavelength : float, units 'm'
        Wavelngth of the beam

    distance : float, units 'm'
        Propagation distance

    paraxial_aprox : bool, default=False
        If True, use paraxial approximation (Fresnel approximation)

    Returns
    -------
    output : xr.DataArray
        DataArray with same dimensions and coordinates but updated data (phase shifted)
    """
    
    #validation: check if first 2 dimensions and coordinates are 'k_x' and 'k_y'
    if da.dims[:2] == ('k_x','k_y'):
        pass
    else:
        raise ValueError(f"dims {da.dims} must be ('k_x','k_y')")
    
    if list(da.coords.keys())[:2] == ['k_x', 'k_y']:        # da.coords[:2] doesn't work
        pass
    else:
        raise ValueError(f"first two coordinates of {da.coords} must be 'k_x' and 'k_y'")

    coord_1 = da.coords['k_x']
    coord_2 = da.coords['k_y']

    # convert units (carefull: handles treats 'rad/m' same as '1/m')
    coord_1_data = rescale_by_units(coord_1.data, coord_1.units, 'rad/m')
    coord_2_data = rescale_by_units(coord_2.data, coord_2.units, 'rad/m')

    # make meshgrid of coordinates
    coord_1_grid, coord_2_grid = np.meshgrid(coord_1_data, coord_2_data, indexing='ij')

    # calculate wavenumber
    k0 = 2 * np.pi / wavelength

    # calculate phase shift: Non-Paraxial (Exact) Angular Spectrum
    if paraxial_aprox == False:
        kz = np.sqrt(k0**2 - coord_1_grid**2 - coord_2_grid**2)
    
    # calculate phase shift: Fresnel Approximation (Paraxial)
    else: 
        kz = k0 - (coord_1_grid**2 + coord_2_grid**2) / (2 * k0)
    
    phase_shift = kz * distance

    # apply phase shift to data
    data_new = da.values * np.exp(1j * phase_shift)

    # create new DataArray with same coordinates and attributes but updated data
    da_new = da.copy(data=data_new)

    # update label
    if label is not None:
        da_new.attrs['label'] = label

    return da_new

# calculate secondary coordinates for reciprocal space (angle and focus size)
def append_secondary_rec_coordinates(da: xr.DataArray, wavelength: float, focal_length: float = None) -> xr.DataArray:
    """
    Short summary line.

    Longer description if needed.

    Parameters
    ----------
    wavelengh : float, units 'm'
        Wavelngth of the beam

    foacal_lengh : float, default='None', units 'm'
        Focal length of the lens, if None, only angle coordinates are calculated

    Returns
    -------
    output : float
        Explanation of the returned value.
    """
    
    #validation: check if first 2 dimensins are 'k_x' and 'k_y'
    if da.dims[:2] == ('k_x','k_y'):
        pass
    else:
        raise ValueError(f"dims {da.dims} must be ('k_x','k_y')")

    coord_1_data = da.coords['k_x'].data
    coord_2_data = da.coords['k_y'].data

    coord_1_units = da.coords['k_x'].units
    coord_2_units = da.coords['k_y'].units

    # convert units (carefull: handles treats 'rad/m' same as '1/m')
    coord_1_data = rescale_by_units(coord_1_data, coord_1_units, 'rad/m')
    coord_2_data = rescale_by_units(coord_2_data, coord_2_units, 'rad/m')

    # calculate angle (Check: conversion factor)
    angle_1 = wavelength * coord_1_data / (2 * np.pi)
    angle_2 = wavelength * coord_2_data / (2 * np.pi)

    # convert to from 'rad' -> 'μrad'
    angle_1 = angle_1 * 1e6
    angle_2 = angle_2 * 1e6

    # append new secondary coordinates
    da = append_coordinate(da,name='θ_x',dim='k_x',data=angle_1,units='μrad')
    da = append_coordinate(da,name='θ_y',dim='k_y',data=angle_2,units='μrad')

    if focal_length is not None:
        # calculate spatial coordinates at focus
        x_1 = focal_length * np.tan(angle_1 * 1e-6) # convert back to rad
        x_2 = focal_length * np.tan(angle_2 * 1e-6)

        # convert to from 'm' -> 'μm'
        x_1 = x_1 * 1e6
        x_2 = x_2 * 1e6

        # append new secondary coordinates
        da = append_coordinate(da,name='x_focus',dim='k_x',data=x_1,units='μm',long_name=r'x_{\text{focus}}')
        da = append_coordinate(da,name='y_focus',dim='k_y',data=x_2,units='μm',long_name=r'y_{\text{focus}}')

    return da

# convert from Intensity to E-Field (ToDo: handle physical units such as 'w/m^2')
def Int_to_Efield(da: xr.DataArray) -> xr.DataArray:
    
    # change name from Intensity to E-Field
    if da.name == 'Intensity':
        name_new = 'E-Field'
        long_name_new = r'E\mathrm{-Field}'
        #print(f"DataArray name 'Intensity' will be updated to '{name_new}' to reflect E-Field.")
    else:
        print(f"Warning: DataArray name is not 'Intensity', but '{da.name}'. Name will not be updated to reflect E-Field.")

    # in case of arbitrary units
    if da.units == 'arb.u.':
        data = np.sqrt(da.values)
    # handle units: if units are 'W/m^2' or 'W/cm^2', convert to 'V/m' (Check: conversion factor)
    else:
        data = np.sqrt(da.values)
        print(f"Don't know how to handle units '{da.units}' for E-Field. Units will not be updated to reflect E-Field.")

    # create new DataArray
    da_new = da.copy(data=data)
    da_new.name = name_new
    da_new.attrs['long_name'] = long_name_new

    return da_new

def Efield_to_Int(da: xr.DataArray) -> xr.DataArray:
    
    # change name from E-Field to Intensity
    if da.name == 'E-Field':
        name_new = 'Intensity'
        #print(f"DataArray name 'E-Field' will be updated to '{name_new}' to reflect Intensity.")
    else:
        print(f"Warning: DataArray name is not 'E-Field', but '{da.name}'. Name will not be updated to reflect Intensity.")

    # in case of arbitrary units
    if da.units == 'arb.u.':
        #data = da.values**2
        data = da.values.real**2 + da.values.imag**2
    # handle units: if units are 'V/m', convert to 'W/m^2' (Check: conversion factor)
    else:
        #data = da.values**2
        data = da.values.real**2 + da.values.imag**2
        print(f"Don't know how to handle units '{da.units}' for Intensity. Units will not be updated to reflect Intensity.")

    # create new DataArray
    da_new = da.copy(data=data)
    da_new.name = name_new
    del da_new.attrs['long_name']

    return da_new

# --- helper functions ---

# helper function to append new coordinates to DataArray (ToDo: decide if to move to utils)
def append_coordinate(da: xr.DataArray, name: str, dim: str, data: np.array, units: str, long_name = None) -> xr.DataArray:
    if long_name is None:
        da = da.assign_coords(
            {name: xr.Variable(dims=dim, data=data, attrs={'units': units})}
        )
    else:
        da = da.assign_coords(
            {name: xr.Variable(dims=dim, data=data, attrs={'units': units, 'long_name': long_name})}
        )
    return da




# test and debugg
if __name__ == "__main__":
    import laser_beam as lb
  
    import matplotlib.pyplot as plt

    # --- Define Beam ---
    beam_nf_I = lb.create_beam_xy(
        type="Gauss",
        label="Near Field",
        name = "Intensity",
        units = "arb.u.",
        func_params={
            'width_x': 1,
            'width_y': 1,
        },
        axis_unit="mm",
        axis_pixelsize=0.05,
        axis_x_N=101,
        axis_y_N=101,
    )


    # --- Calculations ---
    # convert from Intensity to E-Field
    beam_nf_E = Int_to_Efield(beam_nf_I)

    # calculate far-field by fourier transform
    beam_ff_E = fft_xy(beam_nf_E,label="Far-Field")
    # append secondary coordinates to far-field (angle and focus size)
    beam_ff_E = append_secondary_rec_coordinates(beam_ff_E,wavelength=1030e-9,focal_length=100-3)
    
    # convert from E-Field to Intensity
    beam_ff_I = Efield_to_Int(beam_ff_E)
    
    # propagate far-field over distance
    beam_ff_E_2 = propagate(beam_ff_E,wavelength=1030e-9,distance=762e-3)
    
    # calculate near-field by inverse fourier transform
    beam_nf_E_2 = fft_xy(beam_ff_E_2,direction='inverse',label='Back Converted')
    beam_nf_I_2 = Efield_to_Int(beam_nf_E_2)



    # --- Figures ---

    fig,ax = plt.subplots(2, 3,figsize=(12,6))

    lb.plot_2D(beam_nf_I, ax=ax[0,0])
    lb.plot_2D(beam_ff_I, ax=ax[0,1], secondary_axis='angle')
    lb.plot_2D(beam_nf_I_2, ax=ax[0,2])
    
    lb.plot_2D(beam_nf_E, ax=ax[1,0])
    lb.plot_2D(beam_ff_E, ax=ax[1,1], secondary_axis='angle')
    lb.plot_2D(beam_nf_E_2, ax=ax[1,2])


    plt.tight_layout()
    plt.show()


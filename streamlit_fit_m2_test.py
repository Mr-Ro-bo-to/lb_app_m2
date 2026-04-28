import laser_beam as lb

import streamlit as st
import xarray as xr
import matplotlib.pyplot as plt

import tempfile
import os

# helper function to convert edited dataframe back to dataset with metadata
# Convert edited DataFrame back to Dataset, preserving metadata
def df_to_ds_with_meta(edited_df, original_ds):

    # Filter to only active rows, then drop the 'show' column
    edited_df = edited_df[edited_df["show"]].drop(columns=["show"])

    # Identify index columns (dimensions) from the original dataset
    index_cols = list(original_ds.dims)
    
    # Re-set the index to match original structure
    edited_df = edited_df.set_index(index_cols)
    
    # Convert back to Dataset
    new_ds = xr.Dataset.from_dataframe(edited_df)
    
    # Restore global attributes
    new_ds.attrs = original_ds.attrs
    
    # Restore variable attributes
    for var in new_ds.data_vars:
        if var in original_ds:
            new_ds[var].attrs = original_ds[var].attrs
    
    # Restore coordinate attributes
    for coord in new_ds.coords:
        if coord in original_ds.coords:
            new_ds[coord].attrs = original_ds[coord].attrs
    
    return new_ds

# set page layout
st.set_page_config(layout="wide")

st.write("Let's fit M² to some data! 🥳")

# 1) load wideget, 2) header,
col1, col2, col3 = st.columns([2, 1, 2])

# input widgets
with col1:

    # widget for uploading data
    uploaded_file = st.file_uploader("Select data file", 
        type=["xlsx", '.xls'],
        help = "Excel file",
    )
    # if succesfull file uploade, make beam object from file
    if uploaded_file:

        try:
            # Problem: how to use my beam_load() function when using streamlit file uploader
            # Solution: ChatGPT magic. Save uploaded file to a temporary file, get path of that file, hand it to beam_load
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            ds = lb.load_table_to_dataset(tmp_path)
            print("uploading file")

        except Exception as e:
            st.error(f"Error loading data: {e}")
            uploaded_file = None

    # load example data if no file uploaded
    if uploaded_file == None:
        file = 'data\data_M2_example.xlsx'
        ds = lb.load_table_to_dataset(file)

        #print(f"uploaded_file: {ds}")

        #st.dataframe(ds.to_dataframe())

titel = ds.attrs.get("title", None)
date = ds.attrs.get("date", None)
comment = ds.attrs.get("comment", None)

# col1, col2, col3 = st.columns([2, 1, 2])

with col1:
    title = st.text_input("Title", value=titel, disabled=False)
    date = st.text_input("Date", value=date, disabled=False)
    comment = st.text_input("Comment", value=comment, disabled=False)

    ds.attrs["title"] = title
    ds.attrs["date"] = date
    ds.attrs["comment"] = comment

with col3:

    # edit metadata of the dataset
    sub_col1, sub_col2, sub_col3 = st.columns([1, 1, 1])

    with sub_col1:
        wavelength = st.number_input(
            "Wavelength (nm)", value=1030.0, step=1.0
        )
        wavelength = wavelength * 1e-9 # convert to m
        print(f"wavelength: {wavelength} nm")

    # edit metadata of the dataset
    sub_col1, sub_col2, sub_col3 = st.columns([1, 1, 1])

    coord_name = next(iter(ds.coords))

    with sub_col1:
        st.write("x coordinate")
    with sub_col2:
        coord_name_new = st.text_input(f"Labeel:", value=coord_name)
        # Rename and write back to the variable
        ds = ds.rename({coord_name: coord_name_new})
    with sub_col3:
        unit = st.text_input(f"Unit:", value=ds[next(iter(ds.coords))].attrs.get("units", "-"))
        ds[coord_name_new].attrs["units"] = unit

    # Variables
    for i, key in enumerate(ds.data_vars):
        sub_col1, sub_col2, sub_col3 = st.columns([1, 1, 1])
        with sub_col1:
            variable_name = st.text_input(f"Variable:", value=key, key = f"{i} var")
            #if variable_name != key:
            ds = ds.rename({key: variable_name})
            ds[variable_name].attrs["label"] = variable_name
        with sub_col2:
            dimension = st.text_input(f"Label:", value=ds[variable_name].attrs.get("long_name", key), key = f"{i} dim")
            # if dimension != ds[variable_name].attrs.get("long_name", key):
            ds[variable_name].attrs["long_name"] = dimension
                # ds = ds.assign({variable_name: ds[variable_name]})
        with sub_col3:
            unit = st.text_input(f"Unit:", value=ds[variable_name].attrs.get("units", "-"), key = f"{i} unit")
            #if unit != ds[variable_name].attrs.get("units"):
            ds[variable_name].attrs["units"] = unit
                # ds = ds.assign({variable_name: ds[variable_name]})



st.markdown("---") # just a horizontal line

# ds_edit = ds.copy(deep=True)

# # readback changes to metadata
# sub_col1, sub_col2, sub_col3 = st.columns([1, 1, 1])
# with sub_col1:
#     st.write("Coorinate")
# with sub_col2:
#     st.text_input(f"Dimension:", value=coord_name, key = "coord_dim2")
#     # Rename and write back to the variable
#     #ds = ds.rename({coord_name: coord_name_new})
# with sub_col3:
#     st.text_input(f"Unit:", value=ds_edit[next(iter(ds.coords))].attrs.get("units", "-"), key = "coord_unit2")
# # Variables
# for key in list(ds_edit.data_vars):
#     sub_col1, sub_col2, sub_col3 = st.columns([1, 1, 1])
#     with sub_col1:
#         st.text_input(f"Variable:", value=key, key = key + "_var2")
#     with sub_col2:
#         st.text_input(f"Dimension:", value=ds_edit[key].attrs.get("long_name", key), key = key + "_dim2") 
#     with sub_col3:
#         st.text_input(f"Unit:", value=ds_edit[key].attrs.get("units", "-"), key = key + "_unit2")
        
# do some calculations on tthe data

# waist -> radius
ds = ds/2

# get x and y DataArray (assume colum header are 'X' and 'Y')
# get first two variable:
# Get the names of the first two variables


# 2. Display the editor (as we did before)
col1, col2, col3 = st.columns([2, 2, 4])

df = ds.to_dataframe().reset_index()

# Add 'show' column if not present (default True)
if "show" not in df.columns:
    df["show"] = True

# display data editor
with col1:
    df = st.data_editor(
        df, 
        height=500, 
        num_rows="dynamic",
        column_config={
        "Position": st.column_config.NumberColumn(required=True, default=0),
        "X": st.column_config.NumberColumn(required=True, default=0),
        "Y": st.column_config.NumberColumn(required=True, default=0),
        "show": st.column_config.CheckboxColumn(required=True, default=True),
    })
    
ds = df_to_ds_with_meta(df, ds)

var_names = list(ds.data_vars)[:2]
x = ds[var_names[0]]
y = ds[var_names[1]]
x_fit = lb.fit_m2(x, wavelength=wavelength)
y_fit = lb.fit_m2(y, wavelength=wavelength)

# plot the data and the fit
with col3:
    fig, ax = plt.subplots(1, 1)
    lb.plot_1D([x, y, x_fit, y_fit],
        title=ds.attrs['title'],
        plot_styles=lb.STYLE_M2_FIT,
        overlays_label_show=False,
        legend_kwargs= {'loc': 'upper center', 'ncol': 2},
    )
    st.pyplot(fig)





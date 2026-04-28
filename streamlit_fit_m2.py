# import libraries
# my own library (backend, doing the actual work)
import laser_beam as lb

# web app
from scipy import io
import streamlit as st
import pandas as pd


#import xarray as xr

# loading file
import tempfile
import os

# plotting
import matplotlib.pyplot as plt
# import textwrap
import io

# set page layout
st.set_page_config(layout="wide")

# info string for warning of faulty input data and validation
data_info_string = ""

st.write("Let's fit M² to some data! 🥳")

data_source = st.radio(
    "Select data source", 
    options=["Upload file", "Manual input"],
    horizontal=True,
)



# load data, dispaly/edit header
col1, spacer, col2, spacer, col3 = st.columns([2, 0.2, 1, 0.2, 2])

# input widgets
if data_source == "Upload file":
    with col1:

        sub_col1, sub_col3 = st.columns([2, 1])

        with sub_col1:
            # widget for uploading data
            uploaded_file = st.file_uploader("Select data file", 
                type=["xlsx", '.xls'],
                help = "Excel file",
            )
        uploaded_error = False
    # if succesfull file uploade, make beam object from file
    if uploaded_file:

        try:
            # Problem: how to use my beam_load() function when using streamlit file uploader
            # Solution: ChatGPT magic. Save uploaded file to a temporary file, get path of that file, hand it to beam_load
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            input_dict = lb.load_table_to_flat_dict(tmp_path)
        
            #ds = lb.load_table_to_dataset(tmp_path)
            # print("uploading file")
            uploaded_error = False

        except Exception as e:
            st.error(f"Error loading data: {e}")
            uploaded_file = None
            uploaded_error = True

    # load example data if no file uploaded
    if uploaded_file == None:
        file = 'data/data_M2_example.xlsx'
        input_dict = lb.load_table_to_flat_dict(file)

    
    if uploaded_file == None:
        with sub_col3:
            if uploaded_error:
                info_text = "Error loading file - using example data"
                st.error(info_text)
            else:
                info_text = "No file uploaded - using example data"
                st.info(info_text)


elif data_source == "Manual input":
        # load empty template
        file = 'data\data_M2_empty.xlsx'
        input_dict = lb.load_table_to_flat_dict(file)




# extract metadata from input_dict
titel = input_dict.get("title", 'Title')
date = input_dict.get("date", None)
comment = input_dict.get("comment", None)

# laod and validate wavelength
wavelength = input_dict.get("wavelength", None)# default value in m
if wavelength == None:
    wavelength = float(1000e-9)
    data_info_string += f'  \nWavelength not correclty specified. Set to 1000nm'
wavelength = float(wavelength)

columns = input_dict['measurements']
coordinate = columns.pop('Coordinate')

items = list(columns.items())
data_1_key, data_1_dict = items[0]
data_2_key, data_2_dict = items[1]


data_1 = data_1_dict['values']
data_2 = data_2_dict['values']

# display/edit file header
with col1:
    sub_col1, sub_col2, sub_col3 = st.columns([1, 1, 1])

    with sub_col1:
        title = st.text_input("Title", value=titel)
        print(f"title: {title}")
    with sub_col2:
        date = st.text_input("Date", value=date, key = "date")

    # display/edit wavelength
    with sub_col3:
        wavelength = st.number_input(
            "Wavelength (nm)", value=wavelength*1e9, step=1.0
        ) * 1e-9 # convert to m

with col1:
    comment = st.text_input("Comment", value=comment, key = "comment")

    st.markdown("---") # just a horizontal line
# display/edit column header
#with col3:

    # edit metadata of the dataset
    sub_col1, sub_col2, sub_col3 = st.columns([1, 1, 1])


    

    # display/edit coordinate
    sub_col1, sub_col2, sub_col3 = st.columns([1, 1, 1])
    with sub_col1:
        st.write("X Coordinate")

    with sub_col2:
        coord_dim = st.text_input(f"Coordinate dimension:", value=coordinate['dim'])
        if coord_dim.strip() == "":
            coord_dim = "Nope"
            data_info_string += f"   \n Coordinate dimension has to be defined. Set to '{coord_dim}'"
    with sub_col3:
        coord_unit = st.text_input(f"Coordinate unit:", value=coordinate['unit'])
        

    # display/edit data header
    sub_col1, sub_col2, sub_col3, sub_col4 = st.columns([0.5, 0.5, 1, 1])
    with sub_col1:
        variable_1_name = st.text_input(f"Variable 1:", value=data_1_key)

    with sub_col2:
        variable_2_name = st.text_input(f"Variable 2:", value=data_2_key)

    with sub_col3:
        variable_dim = st.text_input(f"Variable dimension:", value=data_1_dict['dim'])
        if variable_dim.strip() == "":
            variable_dim = "Nopey"
            data_info_string += f"   \n Coordinate dimension has to be defined. Set to '{variable_dim}'"
    with sub_col4:
        variable_unit = st.text_input(f"Variable unit:", value=data_1_dict['unit'])

    
# display/edit data



num_points = len(data_1)

data_dict = {
    f'{coord_dim}': coordinate['values'],
    f'{variable_1_name}': data_1,
    f'{variable_2_name}': data_2,
    'show': [True]*num_points, # this column is used to show/hide points in the plot
}





# print(f"data_list: {data_list}")

with col2:

    if data_source == "Manual input":
        height = 400
    elif data_source == "Upload file":
        height = 500



    data_dict = st.data_editor(
        data_dict,
        num_rows="dynamic",
        height=height,
        hide_index=True,
        column_config={
            f'{coord_dim}': st.column_config.NumberColumn(required=True, default=0),
            f'{variable_1_name}': st.column_config.NumberColumn(required=True, default=0),
            f'{variable_2_name}': st.column_config.NumberColumn(required=True, default=0),
            'show': st.column_config.CheckboxColumn(required=True, default=True)
        }
    )

    # --- data validation

    # validate units
    valid_data = True
    try:
        lb.rescale_by_units(1,coord_unit, 'm')  
    except Exception as e:
        # st.warning(f"Error in coordinate unit: {e}")
        data_info_string += f"  \nCan't fit. Coordinate unit '{coord_unit}' is not a valid unit of length."
        valid_data = False

    try:
        lb.rescale_by_units(1,variable_unit, 'm')
    except Exception as e:
        data_info_string += f"  \nCan't fit. Variable unit '{variable_unit}' is not a valid unit of length."
        valid_data = False

    # validate that numeric data:


    
    # valid_data = st.checkbox("valid data", value=False)

    # 1. Load everything into a DataFrame
    df_temp = pd.DataFrame(data_dict)

    # 2. Filter rows where 'Show' is True
    df_filtered = df_temp[df_temp["show"] == True]

    # validate df
    if len(df_filtered) >= 3:
        pass
    else: 
        valid_data = False
        data_info_string += "  \n Can't fit. Too little data"

    # 3. Convert back to a dictionary of lists
    # This gives you: { 'coord': [...], 'var1': [...], 'var2': [...] }
    data_dict = df_filtered.to_dict(orient='list')

# rebuild input_dict from edited data
input_dict = {
    "title": title,
    "date": date,
    "comment": comment,
    "wavelength": wavelength,
    'measurements': {
        'Coordinate': {
            "values": data_dict[f'{coord_dim}'],
            "dim": coord_dim,
            "unit": coord_unit          
        },
        variable_1_name: {
            "values": data_dict[f'{variable_1_name}'],
            "dim": variable_dim,
            "unit": variable_unit
        },
        variable_2_name: {
            "values": data_dict[f'{variable_2_name}'],
            "dim": variable_dim,
            "unit": variable_unit
        }
    }
}

# print(f"Edited input_dict: {input_dict}")

ds = lb.flat_dict_to_dataset(
    input_dict,
    sort = f'{coord_dim}',
)
with col3:

    sub_col1, sub_col2, sub_col3 = st.columns([1, 1, 1])
    with sub_col1:
         waist_to_radius = st.checkbox(
            "Waist -> Radius",
            value=True,
            help="If your data is waist, check this box to convert it radius for M² fitting. The conversion is r = w/2."
            )
    if waist_to_radius:
        ds = ds/2


    with sub_col2:
        show_overlays = st.checkbox("Show overlays", value=False)


# print ("dict: ", input_dict)
# print(f"Dataset: {ds}")


var_names = list(ds.data_vars)[:2]
x = ds[var_names[0]]
y = ds[var_names[1]]
if valid_data:
    x_fit = lb.fit_m2(x, wavelength=wavelength)
    y_fit = lb.fit_m2(y, wavelength=wavelength)

# st.write(ds)
# print(f"Dataset: {ds}")

# plot the data and the fit
with col3:
    fig, ax = plt.subplots(1, 1)
    # lb.plot_1D(ds)

    plot_data = [x, y]
    if valid_data:
        plot_data += [x_fit, y_fit]

    lb.plot_1D(plot_data,
        title=ds.attrs['title'],
        plot_styles=lb.STYLE_M2_FIT,
        overlays_show=show_overlays,
        overlays_label_show=False,
        legend_kwargs= {'loc': 'upper center', 'ncol': 2},
    )

    
    plt.tight_layout()
    st.pyplot(fig)

    sub_col1, sub_col2, = st.columns([1, 1])

    # Save to a buffer (in-memory file)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")

    
    with sub_col1:
        # Trigger download via button
        st.download_button(
            label="Download Plot as PNG",
            data=buf.getvalue(),
            file_name="my_plot.png",
            mime="image/png"
        )

    flat_dict = lb.dataset_to_flat_dict(ds)

     # Trigger download of Excel file
    with sub_col2:
        st.download_button(
            label="Download Excel",
            data=lb.flat_dict_to_excel_bytes(flat_dict),
            file_name="results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

if data_info_string != "":
    st.warning(f"Invalid input data:{data_info_string}")
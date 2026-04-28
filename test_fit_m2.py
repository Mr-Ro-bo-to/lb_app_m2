import laser_beam as lb

import streamlit as st
import matplotlib.pyplot as plt

file = 'data\data_M2.xlsx'
ds = lb.load_table_to_dataset(file)

x = ds['X']
y = ds['Y']

lb.info(x)
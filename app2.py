# import all required libraries
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_digits  # datasets
from sklearn.manifold import TSNE  # t-SNE algorithm
from sklearn.decomposition import PCA  # PCA algorithm
import umap.umap_ as umap  # UMAP algorithm
import plotly.express as px  # for plotting
import time  # to measure computation time

# set the app title
st.title('Dimensionality Reduction Visualisation')

# select the dataset
dataset_name = st.sidebar.selectbox('Select Dataset', ('Iris', 'Digits'))  # user chooses dataset

# load the dataset based on selection
if dataset_name == 'Iris':
    # loading iris dataset
    iris_data = load_iris()
    x_data = iris_data.data
    y_data = iris_data.target
elif dataset_name == 'Digits':
    # loading digits dataset
    digits_data = load_digits()
    x_data = digits_data.data
    y_data = digits_data.target
else:
    st.write('Dataset not found.')
    x_data = None
    y_data = None

# select the algorithm
algorithm = st.sidebar.selectbox('Select Algorithm', ('t-SNE', 'UMAP', 'PCA'))  # user chooses algorithm
n_components = st.sidebar.radio('Number of Components', (2, 3))  # choose number of dimensions

# perform dimensionality reduction based on algorithm
if algorithm == 't-SNE':
    # get perplexity from user
    perplexity_value = st.sidebar.slider('Perplexity', 5, 50, 30)
    # start timer
    start_time = time.time()
    # create t-SNE model
    tsne_model = TSNE(n_components=n_components, perplexity=perplexity_value)
    # fit and transform data
    x_transformed = tsne_model.fit_transform(x_data)
    # end timer
    end_time = time.time()
elif algorithm == 'UMAP':
    # get n_neighbors and min_dist from user
    n_neighbors_value = st.sidebar.slider('Number of Neighbors', 2, 50, 15)
    min_dist_value = st.sidebar.slider('Minimum Distance', 0.0, 1.0, 0.1)
    # start timer
    start_time = time.time()
    # create UMAP model
    umap_model = umap.UMAP(n_neighbors=n_neighbors_value, min_dist=min_dist_value, n_components=n_components)
    # fit and transform data
    x_transformed = umap_model.fit_transform(x_data)
    # end timer
    end_time = time.time()
elif algorithm == 'PCA':
    # start timer
    start_time = time.time()
    # create PCA model
    pca_model = PCA(n_components=n_components)
    # fit and transform data
    x_transformed = pca_model.fit_transform(x_data)
    # end timer
    end_time = time.time()
else:
    st.write('Algorithm not recognized.')
    x_transformed = None

# prepare data for plotting
if x_transformed is not None:
    # create dimension names
    dimension_names = []
    for i in range(n_components):
        dimension_names.append('Dimension ' + str(i+1))
    # create dataframe
    data_frame = pd.DataFrame(x_transformed, columns=dimension_names)
    data_frame['label'] = y_data.astype(str)

    # plot the results
    if n_components == 2:
        # create 2D scatter plot
        figure = px.scatter(data_frame, x='Dimension 1', y='Dimension 2', color='label', title=algorithm + ' Visualization')
        st.plotly_chart(figure)
    elif n_components == 3:
        # create 3D scatter plot
        figure = px.scatter_3d(data_frame, x='Dimension 1', y='Dimension 2', z='Dimension 3', color='label', title=algorithm + ' Visualization')
        st.plotly_chart(figure)
    else:
        st.write('Cannot plot with the selected number of components.')

    # display the time taken
    time_elapsed = end_time - start_time
    st.write('Time taken: {:.2f} seconds'.format(time_elapsed))
else:
    st.write('No data to display.')

# extra comments
# TODO: Add more datasets and algorithms
# Note: This is a simple app for visualizing dimensionality reduction
# The code might have some inefficiencies

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



# Set the app title
st.title('Dimensionality Reduction Visualization')

# Introduction
st.write("""
Welcome! This app lets you explore how different dimensionality reduction algorithms work on real datasets.
""")

# Explain Dimensionality Reduction
st.header('What is Dimensionality Reduction?')
st.write("""
Sometimes, data has many features, which makes it hard to visualize or process. Dimensionality reduction helps simplify the data by reducing the number of features while keeping important patterns.
""")

# Instructions
st.header('How to Use This App')
st.write("""
1. **Select a Dataset**: Choose either the Iris or Digits dataset from the sidebar.
2. **Pick an Algorithm**: Select PCA, t-SNE, or UMAP.
3. **Adjust Parameters**: If available, adjust parameters like the number of components or perplexity.
4. **View the Results**: See how the algorithm transforms the data and observe any patterns.
""")





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
algorithm = st.sidebar.selectbox('Select Algorithm', ('Select','t-SNE', 'UMAP', 'PCA'))  # user chooses algorithm
n_components = st.sidebar.radio('Number of Components', (2, 3))  # choose number of dimensions

# perform dimensionality reduction based on algorithm
if algorithm == 'Select':
    st.write('Please select an algorithm.')
    x_transformed = None
elif algorithm == 't-SNE':
    # get perplexity from user
    
    # Add this code to your Streamlit app to include the t-SNE explanation with equations

    

    perplexity_value = st.sidebar.slider('Perplexity', 5, 50, 30)
    st.write("""
    **Perplexity**: Controls how the algorithm considers neighboring points. Try adjusting it to see how the visualization changes.
    """)
    start_time = time.time()
    # create t-SNE model
    tsne_model = TSNE(n_components=n_components, perplexity=perplexity_value, n_iter=500)
    # fit and transform data
    x_transformed = tsne_model.fit_transform(x_data)
    # end timer
    end_time = time.time()
elif algorithm == 'UMAP':
    # get n_neighbors and min_dist from user
    n_neighbors_value = st.sidebar.slider('Number of Neighbors', 2, 50, 15)
    min_dist_value = st.sidebar.slider('Minimum Distance', 0.0, 1.0, 0.1)
    st.write("""
    **About UMAP**: A fast algorithm that preserves more of the global structure of data compared to t-SNE.

    **Number of Neighbors**: Affects how UMAP balances local versus global structure.

    **Minimum Distance**: Smaller values make clusters tighter.
    """)
    start_time = time.time()
    # create UMAP model
    umap_model = umap.UMAP(n_neighbors=n_neighbors_value, min_dist=min_dist_value, n_components=n_components, n_epochs=200)
    # set n_epchs to 200 to reduce the number of optimization iterations
    # fit and transform data
    x_transformed = umap_model.fit_transform(x_data)
    # end timer
    end_time = time.time()
elif algorithm == 'PCA':
    st.write("""
    **About PCA**: A simple method that reduces dimensions by projecting data onto the directions with the most variance.
    """)
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
 
 
 
 


if algorithm == 't-SNE':
    st.header('t-SNE: Unveiling the Hidden Structure of High-Dimensional Data')

    st.write("""
    *Welcome to the exploration of t-Distributed Stochastic Neighbor Embedding (t-SNE)! This powerful algorithm helps us visualize complex, high-dimensional datasets by reducing them to two or three dimensions. Let's dive into how t-SNE works and uncover the magic behind its ability to reveal patterns in data.*
    """)

    st.subheader('Why Do We Need t-SNE?')

    st.write("""
    Imagine you have a dataset where each data point has dozens or even hundreds of features. Visualizing such high-dimensional data is nearly impossible. t-SNE comes to the rescue by transforming this data into a lower-dimensional space while preserving the meaningful relationships between points.
    """)

    st.subheader('The Core Idea of t-SNE')

    st.write("""
    t-SNE aims to map similar high-dimensional data points to nearby points in a low-dimensional space and dissimilar points to distant ones. It does this by:

    1. **Measuring Pairwise Similarities** between data points in the high-dimensional space.
    2. **Defining Similarities** in the low-dimensional space.
    3. **Minimizing the Difference** between these two sets of similarities.
    """)

    st.subheader('Step 1: Measuring Similarities in High Dimensions')

    st.write('**Understanding Pairwise Similarities**')

    st.write("""
    - For each pair of data points $x_i$ and $x_j$, we compute a similarity measure $p_{ij}$.
    - This similarity reflects how likely it is that $x_i$ would consider $x_j$ as a neighbor.
    """)

    st.write('**Calculating the Similarities**')

    st.write("""
    - We use a Gaussian distribution centered at each point.
    - The conditional probability that $x_i$ picks $x_j$ as a neighbor is:
    """)

    st.latex(r'''
    p_{j|i} = \frac{\exp\left(-\frac{\| x_i - x_j \|^2}{2 \sigma_i^2}\right)}{\sum\limits_{k \ne i} \exp\left(-\frac{\| x_i - x_k \|^2}{2 \sigma_i^2}\right)}
    ''')

    st.write("""
    - $\| x_i - x_j \|$ is the Euclidean distance between $x_i$ and $x_j$.
    - $\sigma_i$ is the bandwidth of the Gaussian for point $x_i$, determined by the perplexity.
    """)

    st.write('The joint probability is symmetrized:')

    st.latex(r'''
    p_{ij} = \frac{p_{j|i} + p_{i|j}}{2N}
    ''')

    st.write('- $N$ is the total number of data points.')

    st.subheader('Step 2: Measuring Similarities in Low Dimensions')

    st.write('**Defining Low-Dimensional Similarities**')

    st.write("""
    - In the low-dimensional space, we represent each high-dimensional point $x_i$ as $y_i$.
    - We define the similarity between $y_i$ and $y_j$ using a Student's t-distribution:
    """)

    st.latex(r'''
    q_{ij} = \frac{\left(1 + \| y_i - y_j \|^2\right)^{-1}}{\sum\limits_{k \ne l} \left(1 + \| y_k - y_l \|^2\right)^{-1}}
    ''')

    st.write('- The heavy tails of the t-distribution help spread out dissimilar points.')

    st.subheader('Step 3: Minimizing the Difference')

    st.write('**Bridging High and Low Dimensions**')

    st.write("""
    - We want the low-dimensional similarities $q_{ij}$ to reflect the high-dimensional similarities $p_{ij}$.
    - We measure the difference using the Kullback-Leibler (KL) divergence:
    """)

    st.latex(r'''
    C = \sum\limits_{i \ne j} p_{ij} \log \left( \frac{p_{ij}}{q_{ij}} \right)
    ''')

    st.write("""
    - Our goal is to minimize $C$ by adjusting the positions of $y_i$ in the low-dimensional space.
    """)

    st.subheader('Optimizing the Positions')

    st.write('**Using Gradient Descent**')

    st.write("""
    - We compute the gradient of the cost function $C$ with respect to $y_i$:
    """)

    st.latex(r'''
    \frac{\partial C}{\partial y_i} = 4 \sum\limits_{j} \left( p_{ij} - q_{ij} \right) \left( y_i - y_j \right) \left( 1 + \| y_i - y_j \|^2 \right)^{-1}
    ''')

    st.write("""
    - We iteratively update $y_i$:
    """)

    st.latex(r'''
    y_i \leftarrow y_i - \eta \frac{\partial C}{\partial y_i}
    ''')

    st.write('- $\eta$ is the learning rate.')

    st.subheader('Understanding Perplexity')

    st.write('**What Is Perplexity?**')

    st.write("""
    - Perplexity controls how we balance attention between local and global data structures.
    - It's a measure of the effective number of neighbors each point has.
    """)

    st.write('**Adjusting Perplexity**')

    st.write("""
    - A low perplexity value focuses on local neighborhoods.
    - A high perplexity value considers a broader range of neighbors.
    - Typical values range from 5 to 50.
    """)

    st.subheader("The Role of the Student's t-Distribution")

    st.write('**Why Not Use a Gaussian Again?**')

    st.write("""
    - In lower dimensions, using a Gaussian can cause the "crowding problem," where distant points end up too close together.
    - The t-distribution has heavier tails, allowing dissimilar points to remain far apart.
    """)

    st.subheader('An Intuitive Analogy')

    st.write("""
    Think of t-SNE as a method for organizing a group photo. In the high-dimensional space, each person has numerous attributes (height, hair color, favorite food). t-SNE helps arrange everyone so that those with similar attributes stand close together, and those who are different stand farther apart.
    """)

    st.subheader('Practical Tips for Using t-SNE')

    st.write("""
    - **Random Initialization**: The algorithm starts with random positions, so results can vary between runs.
    - **Reproducibility**: Set a random seed for consistent results.
    - **Computational Load**: t-SNE can be slow for large datasets. Consider using a subset of data or optimized implementations.
    """)

    st.subheader('Interpreting the Results')

    st.write("""
    - **Clusters**: Groups of points indicate similar data instances.
    - **Distances**: Local distances are meaningful, but global distances may not be.
    - **Axes**: The axes in a t-SNE plot don't have specific meanings; focus on the patterns instead.
    """)

    st.subheader('Limitations of t-SNE')

    st.write("""
    - **Global Structure**: It may distort global relationships to preserve local structures.
    - **Parameter Sensitivity**: Results can change with different perplexity values.
    - **Not for Quantitative Analysis**: t-SNE is best for visualization, not for measuring distances or densities.
    """)

    st.subheader('Experiment with t-SNE in This App')

    st.write('**Adjust the Perplexity**')

    st.write("""
    - Use the slider to change the perplexity value.
    - Observe how the visualization changes.
    - Try different values to see how local and global structures are affected.
    """)

    st.write('**Choose Different Datasets**')

    st.write("""
    - **Iris Dataset**: A smaller dataset with 4 dimensions.
    - **Digits Dataset**: A larger dataset with 64 dimensions.
    - See how t-SNE handles different complexities.
    """)

    st.write('**Have Fun Exploring!**')

    st.write("""
    t-SNE is a fascinating tool that brings high-dimensional data to life. By understanding how it works, you can better interpret the visualizations and gain insights into your data.
    """)
elif algorithm == 'UMAP':
    # Display the UMAP explanation
    st.header('UMAP: Unveiling the Manifold Structure of High-Dimensional Data')

    st.write("""
    *Welcome to the exploration of Uniform Manifold Approximation and Projection (UMAP)! This advanced algorithm helps us visualize complex, high-dimensional datasets by reducing them to two or three dimensions while preserving the underlying manifold structure. Let's dive into how UMAP works and understand the principles behind its ability to reveal patterns in data.*
    """)

    st.subheader('Why Use UMAP?')

    st.write("""
    High-dimensional data can be challenging to visualize and analyze. UMAP provides a fast and effective method for dimensionality reduction that maintains both local and global structures in the data. It's particularly useful for large datasets and can often preserve more of the data's global geometry compared to other techniques like t-SNE.
    """)

    st.subheader('The Core Concepts of UMAP')

    st.write("""
    UMAP is based on manifold learning and topological data analysis. It assumes that the data is uniformly distributed on a Riemannian manifold that can be approximated locally as a fuzzy topological structure. The algorithm involves:

    1. **Constructing a High-Dimensional Graph**: Modeling the data's manifold structure.
    2. **Constructing a Low-Dimensional Graph**: Projecting the manifold into lower dimensions.
    3. **Optimizing the Layout**: Minimizing the difference between the high-dimensional and low-dimensional graphs.
    """)

    st.subheader('Step 1: Constructing the High-Dimensional Graph')

    st.write('**Defining the Fuzzy Simplicial Set**')

    st.write("""
    - For each data point $x_i$, UMAP defines a local neighborhood based on the number of neighbors specified.
    - The **fuzzy membership strength** between points $x_i$ and $x_j$ is calculated using:

    """)
    st.latex(r'''
    \mu_{ij} = \exp\left(-\frac{\max(0, d(x_i, x_j) - \rho_i)}{\sigma_i}\right)
    ''')
    st.write("""
    - $d(x_i, x_j)$ is the distance between points $x_i$ and $x_j$.
    - $\rho_i$ is the distance to the nearest neighbor of $x_i$, ensuring connectivity.
    - $\sigma_i$ is a scaling factor determined by solving:
    """)
    st.latex(r'''
    \sum_{j} \exp\left(-\frac{\max(0, d(x_i, x_j) - \rho_i)}{\sigma_i}\right) = \log_2(k)
    ''')
    st.write("""
    - $k$ is the number of neighbors (controlled by the **n_neighbors** parameter).
    """)

    st.subheader('Step 2: Constructing the Low-Dimensional Graph')

    st.write('**Defining the Low-Dimensional Embedding**')

    st.write("""
    - In the low-dimensional space, we aim to find embeddings $y_i$ that preserve the high-dimensional relationships.
    - The similarity between points $y_i$ and $y_j$ is modeled using a **low-dimensional fuzzy simplicial set** with membership strengths:
    """)
    st.latex(r'''
    v_{ij} = \left(1 + a \| y_i - y_j \|^{2b} \right)^{-1}
    ''')
    st.write("""
    - $\| y_i - y_j \|$ is the Euclidean distance between $y_i$ and $y_j$ in the low-dimensional space.
    - Parameters $a$ and $b$ are determined based on the **min_dist** parameter to control the embedding's tightness.

    """)

    st.subheader('Step 3: Optimizing the Layout')

    st.write('**Minimizing the Cross-Entropy Loss**')

    st.write("""
    - The goal is to make the low-dimensional fuzzy simplicial set as similar as possible to the high-dimensional one.
    - We achieve this by minimizing the cross-entropy between the two sets:
    """)

    st.latex(r'''
    C = \sum_{(i,j) \in E} \mu_{ij} \log \left( \frac{\mu_{ij}}{v_{ij}} \right) + (1 - \mu_{ij}) \log \left( \frac{1 - \mu_{ij}}{1 - v_{ij}} \right)
    ''')

    st.write("""
    - $E$ represents the set of edges (pairs of points) in the high-dimensional graph.
    - The optimization adjusts the positions of $y_i$ to minimize $C$, typically using stochastic gradient descent.
    """)

    st.subheader('Understanding the Parameters')

    st.write('**Number of Neighbors ($n\_neighbors$)**')

    st.write("""
    - Controls the balance between local and global structure preservation.
    - Smaller values focus on local details, while larger values capture more global relationships.
    """)

    st.write('**Minimum Distance ($min\_dist$)**')

    st.write("""
    - Determines how tightly points are clustered together.
    - Smaller values result in denser clusters, preserving more local structure.
    - Larger values spread out the data, emphasizing global structure.
    """)

    st.subheader('An Intuitive Explanation')

    st.write("""
    Imagine unfolding a crumpled piece of paper. UMAP tries to flatten the high-dimensional manifold (the crumpled paper) into a lower-dimensional space without tearing or overlapping, preserving the distances and relationships as much as possible.

    """)

    st.subheader('Advantages of UMAP')

    st.write("""
    - **Speed**: UMAP is generally faster than t-SNE, especially on large datasets.
    - **Preservation of Global Structure**: It often maintains more of the overall data geometry.
    - **Scalability**: Suitable for large-scale data visualization.
    """)

    st.subheader('Practical Tips for Using UMAP')

    st.write("""
    - **Parameter Tuning**: Experiment with **n_neighbors** and **min_dist** to find the best representation for your data.
    - **Reproducibility**: Set a random seed using the **random_state** parameter for consistent results.
    - **Interpretation**: Clusters in the embedding can suggest meaningful groupings, but always validate findings with domain knowledge.
    """)

    st.subheader('Interpreting the Results')

    st.write("""
    - **Clusters**: Groups of points may represent similar data instances or classes.
    - **Distances**: Both local and some global distances are meaningful.
    - **Structure**: Patterns can reveal underlying manifold shapes or data organization.
    """)

    st.subheader('Limitations of UMAP')

    st.write("""
    - **Overinterpretation**: Be cautious not to overinterpret small distances or structures without additional evidence.
    - **Parameter Sensitivity**: Different parameter settings can lead to varying results.
    - **Complexity**: The mathematical foundation is more complex, which might make interpretation less straightforward.
    """)

    st.subheader('Experiment with UMAP in This App')

    st.write('**Adjust the Number of Neighbors**')

    st.write("""
    - Use the slider to change **n_neighbors**.
    - Observe how it affects the balance between local and global structure.
    """)

    st.write('**Adjust the Minimum Distance**')

    st.write("""
    - Modify **min_dist** to see how tightly points are clustered.
    - Smaller values create tighter clusters; larger values spread the data out.
    """)

    st.write('**Choose Different Datasets**')

    st.write("""
    - Test UMAP on the **Iris** and **Digits** datasets.
    - See how it handles datasets with different sizes and complexities.
    """)

    st.write('**Have Fun Exploring!**')

    st.write("""
    UMAP is a powerful tool for visualizing and understanding high-dimensional data. By experimenting with the parameters and datasets, you can gain insights into the structure and relationships within your data.
    """)
elif algorithm == 'PCA':
    # Display the PCA explanation
    st.header('PCA: Simplifying High-Dimensional Data')

    st.write("""
    *Welcome to the exploration of Principal Component Analysis (PCA)! PCA is a fundamental dimensionality reduction technique that transforms high-dimensional data into a lower-dimensional form while retaining as much variability as possible. Let's delve into how PCA works and understand its ability to simplify complex datasets.*
    """)

    st.subheader('Why Use PCA?')

    st.write("""
    High-dimensional data can be difficult to visualize and analyze due to the "curse of dimensionality." PCA helps by reducing the number of dimensions without losing significant information. It does this by finding new axes (principal components) that capture the most variance in the data.
    """)

    st.subheader('The Core Concepts of PCA')

    st.write("""
    PCA involves:
    
    1. **Standardizing the Data**: Ensuring each feature has a mean of zero and unit variance.
    2. **Computing the Covariance Matrix**: Understanding how variables relate to each other.
    3. **Calculating Eigenvectors and Eigenvalues**: Finding the principal components.
    4. **Projecting the Data**: Transforming the original data onto the new axes.
    """)

    st.subheader('Step 1: Standardizing the Data')

    st.write("""
    - Center the data by subtracting the mean of each feature:
    """)

    st.latex(r'''
    x_{ij}^{\text{centered}} = x_{ij} - \bar{x}_j
    ''')

    st.write("""
    - Where \( x_{ij} \) is the value of the \( j \)-th feature for the \( i \)-th sample, and \( \bar{x}_j \) is the mean of the \( j \)-th feature.
    """)

    st.subheader('Step 2: Computing the Covariance Matrix')

    st.write("""
    - The covariance matrix \( \mathbf{C} \) captures the variance and covariance between features:
    """)

    st.latex(r'''
    \mathbf{C} = \frac{1}{n - 1} \mathbf{X}^\top \mathbf{X}
    ''')

    st.write("""
    - Where \( \mathbf{X} \) is the matrix of centered data, and \( n \) is the number of samples.
    """)

    st.subheader('Step 3: Calculating Eigenvectors and Eigenvalues')

    st.write("""
    - Solve the eigenvalue equation:
    """)

    st.latex(r'''
    \mathbf{C} \mathbf{v} = \lambda \mathbf{v}
    ''')

    st.write("""
    - Where:
        - \( \lambda \) are the eigenvalues (scalar values representing the variance along a component).
        - \( \mathbf{v} \) are the eigenvectors (principal components).
    - The eigenvectors are ordered by decreasing eigenvalues.
    """)

    st.subheader('Step 4: Projecting the Data')

    st.write("""
    - The data is projected onto the top \( k \) principal components:
    """)

    st.latex(r'''
    \mathbf{Y} = \mathbf{X} \mathbf{W}
    ''')

    st.write("""
    - Where:
        - \( \mathbf{Y} \) is the transformed data in the lower-dimensional space.
        - \( \mathbf{W} \) is the matrix of the top \( k \) eigenvectors.
    """)

    st.subheader('Interpreting the Results')

    st.write("""
    - **Variance Explained**: Each principal component explains a certain amount of the total variance. The first principal component captures the most variance, the second captures the next most, and so on.
    - **Dimensionality Reduction**: By selecting the top \( k \) principal components, we reduce the dimensionality while retaining most of the variability.
    """)

    st.subheader('Advantages of PCA')

    st.write("""
    - **Simplicity**: PCA is straightforward to compute and understand.
    - **Noise Reduction**: By removing less significant components, we can reduce noise.
    - **Visualization**: Reducing data to 2 or 3 dimensions allows for visualization of high-dimensional data.
    """)

    st.subheader('Limitations of PCA')

    st.write("""
    - **Linearity**: PCA assumes linear relationships and may not capture complex nonlinear structures.
    - **Scaling Sensitivity**: The results can be affected by the scaling of features; standardization is essential.
    - **Interpretability**: Principal components are linear combinations of original features and may not have direct interpretations.
    """)

    st.subheader('Experiment with PCA in This App')

    st.write("""
    - **Number of Components**: Choose 2 or 3 components to see how the data looks in reduced dimensions.
    - **Variance Explained**: Observe how much variance is captured by the selected components.
    - **Datasets**: Try PCA on different datasets to see how it performs.
    """)

    st.write('**Have Fun Exploring!**')

    st.write("""
    PCA is a fundamental tool in data analysis and machine learning. By experimenting with PCA in this app, you can gain insights into the structure of your data and the importance of different features.
    """)

    # Start timer
    start_time = time.time()

    # Create and fit the PCA model
    pca_model = PCA(n_components=n_components)
    x_transformed = pca_model.fit_transform(x_data)

    # End timer
    end_time = time.time()

    # Calculate the explained variance ratio
    explained_variance = pca_model.explained_variance_ratio_

    # Display the explained variance
    st.write(f"Explained Variance Ratio of the selected components: {explained_variance.sum():.2f}")
    
   
st.write("""
Feel free to experiment with different parameters and see how they affect the visualization. This can help you understand how each algorithm works.
""")


# extra comments
# TODO: Add more datasets and algorithms
# Note: This is a simple app for visualizing dimensionality reduction
# The code might have some inefficiencies

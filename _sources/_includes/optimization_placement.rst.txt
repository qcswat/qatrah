Optimization placement module
=============================

In this script, a class called `SensorPlacement` is defined to solve the sensor placement problem, which is a network optimization problem. The script also includes several functions to convert the network data into dataframes, add attributes to the network, generate different factors such as centrality factor and accessibility factor, and plot the network.

Here is a brief description of each function in the script:

1. `convert_oopnet_junctions_to_df(net)`: Converts the junctions in an oopnet network to a dataframe.
2. `convert_oopnet_nodes_to_df(net, default_elevation=0, default_demand=0)`: Converts the nodes in an oopnet network to a dataframe with default elevation and demand values.
3. `drop_y(df)`: Drops columns with a '_y' suffix from the dataframe.
4. `add_attributes(G, nodes)`: Adds attributes such as xcoord, ycoord, demand, and elevation to a graph from a dataframe of nodes.
5. `generate_centrality_factor(G, edges_attr)`: Generates the centrality factor for each edge in the graph.
6. `diameter_length_factor(edges_attr)`: Generates the diameter-length factor for each edge in the graph.
7. `create_edge_weight(edges_attr, A, B)`: Creates the edge weight for each edge in the graph based on the input parameters A and B.
8. `generate_accessibility_factor(edges_attr, P)`: Generates the accessibility factor for each edge in the graph based on the input probability P.
9. `plot_network(G, plot_name="Sensors graph")`: Plots the network.

The `SensorPlacement` class has methods to run different optimization algorithms such as QAOA and VQE, as well as classical optimization algorithms to solve the sensor placement problem. The class also has a `SensorPlacementResults` class to manage the optimization results.

.. automodule:: optimization_placement
   :members:
   :undoc-members:
   :show-inheritance:
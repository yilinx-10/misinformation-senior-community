import mesa
import math
import solara
import networkx as nx
import numpy as np
import matplotlib as plt
from matplotlib.figure import Figure
from model import MisinformationNetwork
from mesa.visualization import Slider, SolaraViz, make_plot_component
from mesa.visualization.utils import update_counter

# Define model parameters
model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "num_residents": Slider(
        label="Number of residents",
        value=20,
        min=10,
        max=100,
        step=10,
    ),
    "staff_resident_ratio": Slider(
        label="Staff Resident Ratio",
        value=0.1,
        min=0.1,
        max=1.0,
        step=0.1,        
    ),
    "avg_node_degree": Slider(
        label="Avg Node Degree",
        value=5,
        min=5,
        max=20,
        step=1,
    ),
    "network_type": {
        "type": "Select",
        "value": "uniform weight",
        "values": ["uniform weight", "random weight", "smallworld"],
        "label": "Network Type",
    },
    "alpha_cognitive": Slider(
        label="Cognitive Ability Alpha",
        value=3,
        min=1,
        max=5,
        step=1,
    ),
    "alpha_dl": Slider(
        label="Digital Literacy Alpha",
        value=3,
        min=1,
        max=5,
        step=1,
    ),
    "seed_mode": {
        "type": "Select",
        "value": "random",
        "values": ["random", "staff", "high_betweenness", "high_degree", "peripheral_betweenness", "peripheral_degree"],
        "label": "Initial Misinformation Drop Mode",
    },
    "fact_checking_prob": Slider(
        label="Fact Checking Probability",
        value=0.05,
        min=0.0,
        max=1.0,
        step=0.05,        
    ),
    "confidence_deprecation_rate": Slider(
        label="Confidence Deprecation Rate",
        value=0.1,
        min=0.1,
        max=1.0,
        step=0.1,        
    ),
}

# Create custom figure to plot the network graph (used in order to plot weighted edges)
@solara.component
def NetPlot(model):
    # Set this to update every turn, define it as mpl figure
    update_counter.get()
    fig = Figure()
    ax = fig.subplots()
    # Get list of colors for each node based on dictionary
    values = [model.grid.get_cell_list_contents([node])[0].belief_scale for node in model.G.nodes()]
    values = np.clip(values, -1, 1)
    # Draw network graph based on colors defined here and node positions/edge weights defined in the model
    cmap = plt.colormaps['coolwarm']
    node_colors = [cmap((value + 1) / 2) for value in values]
    nx.draw(model.G,
            ax=ax,
            pos = model.position,
            labels = nx.get_node_attributes(model.G, "identity"),
            node_size = 300,
            arrowstyle = '->',
            arrowsize = 5,
            font_size = 8, 
            node_color=node_colors,
            cmap = cmap,
            width = model.weight_lst)
    # Plot the figure
    solara.FigureMatplotlib(fig)

@solara.component
def ProportionPlot(model):
    # Set this to update every turn, define it as mpl figure
    update_counter.get()
    fig = Figure()
    ax = fig.subplots()
    data = model.datacollector.get_model_vars_dataframe()
    trust = data["Trust"]
    neglect = data["Neglect"]
    x = data.index
    ax.fill_between(x, 0, trust, label="Trust", color="skyblue")
    ax.fill_between(x, trust, trust + neglect, label="Neglect", color="lightgray")
    ax.fill_between(x, trust + neglect, 1, label="Distrust", color="salmon")
    solara.FigureMatplotlib(fig)

BS_Plot = make_plot_component({"Belief Scores": '#800080'})

# Initialize model instance
model1 = MisinformationNetwork()

# Define page components
page = SolaraViz(
    model1,
    components=[
        NetPlot,
        BS_Plot,
        ProportionPlot,
    ],
    model_params=model_params,
    name="Misinformation Model",
)
# Return page
page 
import math
import random
import networkx as nx
import mesa
from mesa import Model
from agents import NetworkAgent
# Staff and Resident, 
# all receives and are susceptible to misinformation, staff agent higher prob 
# to be active on social media(higher frequency of contact with misinfo)
# but lower susceptability(determined by two broad categories of factors:
# agent-related and interaction. Agent related: socially active, knowledge, digital literacy
# etc. Interaction: from who receives the message tie weight. ). Staff agent
# actively seeks fact-checking services(with prob determine misinfo or not) and 
# actively reach out to residents in contact to clarify info(or perhaps spread truth?). 

# Do I want to allow agents(residents+staff) to form/break ties in the process?

# POTENTIAL CHANGES: higher weight tie tend to have higher weight reciprocity?

# Helper Functions for Network Construction
def remove_self_loops(graph):
    '''
    Self-Loop Remover
    '''
    self_loops = list(nx.selfloop_edges(graph))
    graph.remove_edges_from(self_loops)
    return graph

def staff_or_resident(graph, n, ratio):
    '''
    Assign Staff or Resident attribute to nodes
    '''
    # Calculate Degree Centrality of the Nodes
    # Set weight=None as we care more about 
    # connected or not than strength of connection in this case 
    centrality = nx.degree_centrality(graph)
    # Sort by betweenness centrality
    # randomly select among nodes of high betweenness centrality to be staffs
    selection_range = n * ratio * 2
    sorted_nodes = sorted(centrality, key=centrality.get, reverse=True)[:selection_range:]
    selected_nodes = random.sample(sorted_nodes, n * ratio)

    # Assign node types: Resident or Staff based on betwmess 
    identity = {}
    for node in graph.nodes():
        if node in selected_nodes:
            identity[node] = "Staff"
        else:
            identity[node] = "Resident"

    # Add the node types as an attribute to the graph
    nx.set_node_attributes(graph, identity, "identity")

def drop_misinformation(graph, n, seed_mode):
    '''
    Assign Misinformation to One Node
    '''
    percentile_marker = math.ceil(n * 0.25)
    betweenness_centrality = calc_betweenness(graph)
    degree_centrality = calc_degree(graph)
    betweenness_sort = sorted(betweenness_centrality, key=betweenness_centrality.get, reverse=True)
    degree_sort = sorted(degree_centrality, key=degree_centrality.get, reverse=True)

    if seed_mode == "high_betweenness":
        sorted_nodes = betweenness_sort[:percentile_marker:]
        selection_range = [node for node in graph.nodes() if node['identity'] == 'Resident' and node in sorted_nodes]
    elif seed_mode == "high_degree":
        sorted_nodes = degree_sort[:percentile_marker:]
        selection_range = [node for node in sorted_nodes if node['identity'] == 'Resident' and node in sorted_nodes]
    elif seed_mode == "peripheral_betweenness":
        sorted_nodes = betweenness_sort[n - percentile_marker - 1::]
        selection_range = [node for node in graph.nodes() if node['identity'] == 'Resident' and node in sorted_nodes]
    elif seed_mode == "peripheral_degree":
        sorted_nodes = degree_sort[n - percentile_marker - 1::]
        selection_range = [node for node in graph.nodes() if node['identity'] == 'Resident' and node in sorted_nodes]
    elif seed_mode == "staff":
        selection_range = [node for node in graph.nodes() if node['identity'] == 'Staff']
    elif seed_mode == "random":
        selection_range = list(graph)
    
    seed_node = random.sample(selection_range)
    seed_node.belief_scale = 1

# Helper function for DataCollector
def calc_density(model):
    return nx.density(model.G)

def calc_belief(model):
    belief_scores = 0
    for a in model.grid.get_all_cell_contents():
        belief_scores += a.belief_scale
    return belief_scores

def calc_betweenness(graph):
    return nx.betweenness_centrality(graph, k=None, normalized = True, weight = None, endpoints=False)

def calc_degree(graph):
    return nx.degree_centrality(graph)


class MisinformationnNetwork(Model):
    # Define initiation
    def __init__(
        self,
        num_residents=20,
        avg_node_degree=3,
        network_type="unweighted",
        staff_resident_ratio = 0.1,
        alpha_cognitive = 3, # used to generate beta distribution for cognitive ability
        beta_cognitive = 5,
        alpha_dl = 3, # used to generatebeta distribution for digital literacy
        beta_dl = 5,
        seed_mode = "random",
        fact_checking_prob = 0.95,
        confidence_deprecation_rate = 0.95,
        seed=None,
    ):
        super().__init__(seed=seed)
        random.seed(seed)
        self.fact_checking_prob = fact_checking_prob
        self.confidence_deprecation_rate = confidence_deprecation_rate

        # Set up network: number of nodes, base probability of connection, type of network (binary or weighted)
        self.num_nodes = num_residents * (1 + staff_resident_ratio)
        self.network_type = network_type
        prob = avg_node_degree / self.num_nodes
        # Set up unweighted directed network, creating edges randomly and setting all edge weights to one
        if self.network_type == "unweighted":
            self.G = nx.gnp_random_graph(n=self.num_nodes, p=prob, directed = True)
            self.G = remove_self_loops(self.G)
            for u, v in self.G.edges():
                self.G.edges[u, v]['weight'] = 1              
        # Weighted directed network. weight represents level of trust. 
        # u->v: exist: have contact; high weight: strong investment in this connection
        elif self.network_type == "weighted":
            self.G = nx.gnp_random_graph(n=self.num_nodes, p=prob, directed = True)
            self.G = remove_self_loops(self.G)
            for u, v in self.G.edges():
                # POTENTIAL CHANGES: higher weight tie tend to have higher weight reciprocity?
                self.G.edges[u, v]['weight'] = self.random.random()   
        # A connectedSmall World Network where each individual has ties with its k nearest neightbors
        # Note this example network is undirected  
        elif self.network_type == "smallworld":
            self.G = nx.connected_watts_strogatz_graph(n=self.num_nodes, k = avg_node_degree, p=prob)
        else:
            raise ValueError("Unsupported network type, please select: unweighted, weighted, or smallworld")
        
        # Assign attribute
        staff_or_resident(self.G, self.num_nodes, staff_resident_ratio)

        # Get edge weights
        self.weight_lst = self.get_weight_lst()

        # Get position
        if self.network_type == "smallworld":
            self.position = nx.circular_layout(self.G)
        else:
            self.position = nx.spring_layout(self.G, k = 1, seed=seed)

        # Create grid from network object
        self.grid = mesa.space.NetworkGrid(self.G)
        
        # Define data collection
        self.datacollector = mesa.DataCollector(
            {
                "Trust": lambda m: len(
                    [a for a in m.agents if a.belief_scale > 0]),
                "Distrust": lambda m: len(
                    [a for a in m.agents if a.belief_scale < 0]),
                "Belief Scores": calc_belief,
                "Netsork Density": calc_density,
            }
        )

        # Create Agents and Place them on Grid
        for node in self.G.nodes():
            if node['identity'] == 'Staff':
                agent = NetworkAgent(
                    self,
                    cognitive_ability = random.betavariate(beta_cognitive, alpha_cognitive),
                    digital_literacy = random.betavariate(beta_dl, alpha_dl),
                    se_motivated = False,
                    belief_scale = 0,
                )
                # Add the agent to the node
                self.grid.place_agent(agent, node)
            else:
                agent = NetworkAgent(
                    self,
                    cognitive_ability = random.betavariate(alpha_cognitive, beta_cognitive),
                    digital_literacy = random.betavariate(beta_dl, alpha_dl),
                    se_motivated = True,
                    belief_scale = 0,
                )
                # Add the agent to the node
                self.grid.place_agent(agent, node)

        # Launch Misinformation
        drop_misinformation(self.G, self.num_nodes, seed_mode)

        self.running = True
        self.datacollector.collect(self)

    def get_weight_lst(self):
        if self.network_type == "weighted":
            self.weights = [data['weight'] for _, _, data in self.G.edges(data=True)]
        else:
            self.weights = [1] * len(self.G.edges)

    # Agents take a step, then data is collected
    def step(self):
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)
import math
import random
import networkx as nx
import mesa
from mesa import Model
from agents import NetworkAgent

# Helper Functions for Network Construction
def reciprocated_directed_graph(n, p, seed=None):
    # Generate an undirected Erdos-Renyi graph
    undirected_graph = nx.erdos_renyi_graph(n=n, p=p, seed=seed, directed=False)
    
    # Create a directed graph
    G = nx.DiGraph()
    G.add_nodes_from(undirected_graph.nodes())

    # For each undirected edge, add both (u,v) and (v,u)
    for u, v in undirected_graph.edges():
        G.add_edge(u, v)
        G.add_edge(v, u)

    return G

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
    # Sort by centrality
    # randomly select among nodes of high centrality to be staffs
    # to replicate real-world situations
    selection_range = int(n * ratio * 2)
    sorted_nodes = sorted(centrality, key=centrality.get, reverse=True)[:selection_range:]
    selected_nodes = random.sample(sorted_nodes, int(n * ratio))

    # Assign node types: Resident or Staff based on betwmess 
    identity = {}
    for node in graph.nodes():
        if node in selected_nodes:
            identity[node] = "S"
        else:
            identity[node] = "R"

    # Add the node types as an attribute to the graph
    nx.set_node_attributes(graph, identity, "identity")

def drop_misinformation(model, n, seed_mode):
    '''
    Assign Misinformation to One Node
    '''
    percentile_marker = math.ceil(n * 0.25)
    betweenness_centrality = calc_betweenness(model.G)
    degree_centrality = calc_degree(model.G)
    betweenness_sort = sorted(betweenness_centrality, key=betweenness_centrality.get, reverse=True)
    degree_sort = sorted(degree_centrality, key=degree_centrality.get, reverse=True)

    if seed_mode == "high_betweenness":
        sorted_nodes = betweenness_sort[:percentile_marker:]
        selection_range = [node for node in model.G.nodes() if model.G.nodes[node]['identity'] == 'R' and node in sorted_nodes]
    elif seed_mode == "high_degree":
        sorted_nodes = degree_sort[:percentile_marker:]
        selection_range = [node for node in sorted_nodes if model.G.nodes[node]['identity'] == 'R' and node in sorted_nodes]
    elif seed_mode == "peripheral_betweenness":
        sorted_nodes = betweenness_sort[n - percentile_marker - 1::]
        selection_range = [node for node in model.G.nodes() if model.G.nodes[node]['identity'] == 'R' and node in sorted_nodes]
    elif seed_mode == "peripheral_degree":
        sorted_nodes = degree_sort[n - percentile_marker - 1::]
        selection_range = [node for node in model.G.nodes() if model.G.nodes[node]['identity'] == 'R' and node in sorted_nodes]
    elif seed_mode == "staff":
        selection_range = [node for node in model.G.nodes() if model.G.nodes[node]['identity'] == 'S']
    elif seed_mode == "random":
        selection_range = list(model.G)
    
    seed_node = random.sample(selection_range, k = 1)
    model.grid.get_cell_list_contents(seed_node)[0].belief_scale = 1

# Helper function for DataCollector
def calc_belief(model):
    belief_scores = 0
    for a in model.grid.get_all_cell_contents():
        belief_scores += a.belief_scale
    return belief_scores

def calc_betweenness(graph):
    return nx.betweenness_centrality(graph, k=None, normalized = True, weight = None, endpoints=False)

def calc_degree(graph):
    return nx.degree_centrality(graph)


class MisinformationNetwork(Model):
    # Define initiation
    def __init__(
        self,
        num_residents = 50,
        avg_node_degree = 10,
        network_type="uniform weight",
        staff_resident_ratio = 0.1,
        alpha_cognitive = 3, # used to generate beta distribution for cognitive ability
        beta_cognitive = 5, #fixed
        alpha_dl = 3, # used to generatebeta distribution for digital literacy
        beta_dl = 5, #fixed
        seed_mode = "random",
        fact_checking_prob = 0.05,
        confidence_deprecation_rate = 0.1,
        seed=None,
    ):
        super().__init__(seed=seed)
        random.seed(seed)
        self.fact_checking_prob = fact_checking_prob
        self.confidence_deprecation_rate = confidence_deprecation_rate

        # Set up network: number of nodes, base probability of connection, type of network (binary or weighted)
        self.num_nodes = int(num_residents * (1 + staff_resident_ratio))
        self.network_type = network_type
        prob = avg_node_degree / self.num_nodes
        # Set up unweighted directed network, creating edges randomly and setting all edge weights to one
        if self.network_type == "uniform weight":
            self.G = reciprocated_directed_graph(n=self.num_nodes, p=prob, seed=seed)
            self.G = remove_self_loops(self.G)
            for u, v in self.G.edges():
                self.G.edges[u, v]['weight'] = 1              
        # Weighted directed network. weight represents level of trust. 
        # u->v: exist: have contact; high weight: strong investment in this connection
        elif self.network_type == "random weight":
            self.G = reciprocated_directed_graph(n=self.num_nodes, p=prob, seed=seed)
            self.G = remove_self_loops(self.G)
            for u, v in self.G.edges():
                # POTENTIAL CHANGES: higher weight tie tend to have higher weight reciprocity?
                self.G.edges[u, v]['weight'] = self.random.random()   
        # A connectedSmall World Network where each individual has ties with its k nearest neightbors
        # Note this example network is undirected  
        elif self.network_type == "smallworld":
            self.G = nx.connected_watts_strogatz_graph(n=self.num_nodes, k = avg_node_degree, p=prob)
            for u, v in self.G.edges():
                self.G.edges[u, v]['weight'] = 1  

        # Assign attribute
        staff_or_resident(self.G, self.num_nodes, staff_resident_ratio)

        # Get edge weights
        self.weight_lst = self.get_weight_lst()
        
        self.position = nx.circular_layout(self.G)

        # Create grid from network object
        self.grid = mesa.space.NetworkGrid(self.G)
        
        # Define data collection
        self.datacollector = mesa.DataCollector(
            model_reporters = {
                "Trust": lambda m: len([a for a in m.agents if a.belief_scale > 0.33]) / len(m.agents),
                "Distrust": lambda m: len([a for a in m.agents if a.belief_scale < -0.33]) / len(m.agents),
                "Neglect": lambda m: len([a for a in m.agents if -0.33 <= a.belief_scale <= 0.33]) / len(m.agents),
                "Belief Scores": calc_belief,
            }
        )

        # Create Agents and Place them on Grid
        for node in self.G.nodes():
            if self.G.nodes[node]['identity'] == 'S':
                agent = NetworkAgent(
                    self,
                    node = node,
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
                    node = node,
                    cognitive_ability = random.betavariate(alpha_cognitive, beta_cognitive),
                    digital_literacy = random.betavariate(beta_dl, alpha_dl),
                    se_motivated = True,
                    belief_scale = 0,
                )
                # Add the agent to the node
                self.grid.place_agent(agent, node)

        # Launch Misinformation
        drop_misinformation(self, self.num_nodes, seed_mode)

        self.running = True
        self.datacollector.collect(self)

    def get_weight_lst(self):
        if self.network_type == "random weight":
            self.weights = [data['weight'] for _, _, data in self.G.edges(data=True)]
        else:
            self.weights = [1] * len(self.G.edges)

    # Agents take a step, then data is collected
    def step(self):
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)
        #Stopping Condition
        data = self.datacollector.get_model_vars_dataframe()
        score = abs(data['Belief Scores'].iloc[-1])
        if score >= self.num_nodes - 1:
            self.running = False
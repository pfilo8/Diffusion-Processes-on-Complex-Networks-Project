import random
import networkx as nx

VOTE = 'vote'


def diffusion_of_innovation_model_on_graph(g: nx.Graph, p: float, f: float, q: int, h: float, t: int = 1000):
    """
    g - graph
    p - independence - probability spinson acts independently
    f - flexibility - probability spinson will change opinion if he is independent
    q - number of spinsons in panel
    h - advertising - probability spinson response to advertise
    t - time of simulation
    """
    N = len(g.nodes())
    g = init_opinion(g)

    history = [get_network_opinions(g)]

    for _ in range(t):
        for _ in range(N):
            spinson = get_random_spinson(g)

            if random.random() < p:
                if random.random() < f:
                    g = set_node_opinion(g, spinson, -get_node_opinion(g, spinson))
            else:
                neighbours = get_neighbors(g, spinson)
                panel = get_panel(neighbours, q)
                panel_opinions = get_panel_opinions(g, panel)

                if all_spinsons_have_same_opinion(panel_opinions):
                    g = set_node_opinion(g, spinson, panel_opinions[0])
                else:
                    if random.random() < h:
                        g = set_node_opinion(g, spinson, 1)
        history.append(get_network_opinions(g))
    return history


def init_opinion(G):
    votes = {node: -1 for node in G}
    nx.set_node_attributes(G, votes, VOTE)
    return G


def get_random_spinson(G):
    return random.choice(list(G.nodes))


def get_neighbors(G, node):
    return [v for v in G.neighbors(node)]


def get_node_opinion(G, node):
    return G.nodes[node][VOTE]


def get_panel(neighbours, q):
    return random.sample(neighbours, k=q) if q <= len(neighbours) else neighbours


def get_panel_opinions(G, panel):
    return [get_node_opinion(G, node) for node in panel]


def get_network_opinions(G):
    return nx.get_node_attributes(G, VOTE)


def set_node_opinion(G, node, opinion):
    G.nodes[node][VOTE] = opinion
    return G


def all_spinsons_have_same_opinion(panel_opinions):
    return len(set(panel_opinions)) == 1

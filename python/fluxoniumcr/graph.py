import matplotlib.axes
import matplotlib.pyplot as plt
import networkx as nx


def create_rotated_surface_code(distance: int) -> nx.Graph:
    G = nx.Graph()

    for i in range(distance):
        for j in range(distance):
            # Data qubits are assigned as control qubits.
            G.add_node((i, j), color=0)

    for i in range(distance+1):
        for j in range(distance+1):
            node = (i-0.5, j-0.5)
            color = 1 + i%2 + 2*(j%2)
            if i == 0:
                if j%2 == 1 and j < distance:
                    G.add_node(node, color=color)
                    G.add_edge((i, j-1), node)
                    G.add_edge(node, (i, j))
                else:
                    pass
            elif i == distance:
                if (j + distance)%2 == 1 and 0 < j < distance:
                    G.add_node(node, color=color)
                    G.add_edge((i-1, j-1), node)
                    G.add_edge(node, (i-1, j))
                else:
                    pass
            elif j == 0:
                if i%2 == 0 and i < distance:
                    G.add_node(node, color=color)
                    G.add_edge((i-1, j), node)
                    G.add_edge(node, (i, j))
                else:
                    pass
            elif j == distance:
                if (i + distance)%2 == 0 and 0 < i < distance:
                    G.add_node(node, color=color)
                    G.add_edge((i-1, j-1), node)
                    G.add_edge(node, (i, j-1))
                else:
                    pass
            else:
                G.add_node(node, color=color)
                G.add_edge((i-1, j-1), node)
                G.add_edge((i-1, j), node)
                G.add_edge((i, j-1), node)
                G.add_edge((i, j), node)

    return G


def create_hexagon_code(distance: int) -> nx.Graph:
    G = nx.Graph()

    for i in range(distance):
        for j in range(distance):
            # Data qubits are assigned as control qubits.
            G.add_node((i, j), color=0)

    for i in range(distance):
        for j in range(distance):
            node = (i-0.5, j-0.5)
            color = 1 + (i - j)%3
            if i == 0:
                if j > 0:
                    G.add_node(node, color=color)
                    G.add_edge((i, j-1), node)
                    G.add_edge(node, (i, j))
                else:
                    pass
            elif j == 0:
                if i > 0:
                    G.add_node(node, color=color)
                    G.add_edge((i-1, j), node)
                    G.add_edge(node, (i, j))
                else:
                    pass
            else:
                G.add_node(node, color=color)
                G.add_edge((i-1, j), node)
                G.add_edge((i, j-1), node)
                G.add_edge((i, j), node)

    return G


def create_heavy_hexagon_code(distance: int) -> nx.Graph:
    G = nx.Graph()

    for i in range(distance):
        for j in range(distance):
            # Data qubits are assigned as control qubits.
            G.add_node((i, j), color=0)

    for i in range(distance+1):
        for j in range(distance+1):
            node = (i-0.5, j-0.5)
            if i == 0:
                if j%2 == 0 and 0 < j < distance:
                    flag1 = (i-0.5, j)
                    flag2 = (i-0.5, j-1)
                    G.add_node(node, color=0)
                    G.add_node(flag1, color=1)
                    G.add_node(flag2, color=2)

                    G.add_edge((i, j), flag1)
                    G.add_edge((i, j-1), flag2)
                    G.add_edge(flag1, node)
                    G.add_edge(flag2, node)
                else:
                    pass
            elif i == distance:
                if (j + distance)%2 == 0 and 0 < j < distance:
                    flag1 = (i-0.5, j)
                    flag2 = (i-0.5, j-1)
                    G.add_node(node, color=0)
                    G.add_node(flag1, color=1)
                    G.add_node(flag2, color=2)

                    G.add_edge((i-1, j), flag1)
                    G.add_edge((i-1, j-1), flag2)
                    G.add_edge(flag1, node)
                    G.add_edge(flag2, node)
                else:
                    pass
            elif j == 0:
                if i%2 == 0 and i < distance:
                    node = (i-0.5, j)
                    G.add_node(node, color=1)
                    G.add_edge((i-1, j), node)
                    G.add_edge(node, (i, j))
                else:
                    pass
            elif j == distance:
                if (i + distance)%2 == 0 and 0 < i < distance:
                    node = (i-0.5, j-1)
                    G.add_node(node, color=2)
                    G.add_edge((i-1, j-1), node)
                    G.add_edge(node, (i, j-1))
                else:
                    pass
            elif(i+j)%2 == 0:
                flag1 = (i-0.5, j)
                flag2 = (i-0.5, j-1)
                G.add_node(node, color=0)
                G.add_node(flag1, color=1)
                G.add_node(flag2, color=2)

                G.add_edge((i-1, j), flag1)
                G.add_edge((i, j), flag1)
                G.add_edge((i-1, j-1), flag2)
                G.add_edge((i, j-1), flag2)
                G.add_edge(flag1, node)
                G.add_edge(flag2, node)

    return G


def plot_graph(G: nx.Graph, ax: matplotlib.axes.Axes|None = None) -> None:
    if ax is None:
        ax = plt.gca()

    node_colors = nx.get_node_attributes(G, 'color')

    for i, j in G.nodes:
        color_index = node_colors[(i, j)]
        ax.scatter(i, j, color=f"C{color_index}")

    for (i1, j1), (i2, j2) in G.edges:
        ax.plot([i1, i2], [j1, j2], c='black', zorder=-1)

    ax.set_aspect(1)

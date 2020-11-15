#!/usr/bin/env python
# _*_ coding: utf-8 _*_

__author__ = 'Micah Cearns'
__contact__ = 'micahcearns@gmail.com'
__date__ = 'November 2020'

"""

Network of 1259797 people from wikipedia, with name, links to other people 
and date of birth. Birth dates range from 1830 to 2003. Month and day of month
included for 643915 people, otherwise just the birth year.

"""

import numpy as np
import pandas as pd
import json
import os
import networkx as nx
import nxviz as nv
import matplotlib.pyplot as plt

data_path = ('/Users/MicahJackson/anaconda/Pycharm_WD/'
             'Wikipedia_Network_Analysis/Data/Wikipedia_Raw.json')


def pandas_config():
    """

    Pandas configuration

    :return: Configured Pandas
    """
    options = {
        'display': {
            'max_columns': None,
            'max_colwidth': 25,
            'expand_frame_repr': False,  # Don't wrap to multiple pages
            'max_rows': 14,
            'max_seq_items': 50,  # Max length of printed sequence
            'precision': 10,  # Updating for degree centrality
            'show_dimensions': False},  # Controls SettingWithCopyWarning
        'mode': {
            'chained_assignment': None
        }
    }

    for category, option in options.items():
        for op, value in option.items():
            pd.set_option(f'{category}.{op}', value)

    return


if __name__ == '__main__':

    # ======================================================================
    # Degree centrality
    # ======================================================================
    # Degree centrality. Who has the most neighbours compared to all possible
    # neighbours in the dataset?
    deg_cent = nx.degree_centrality(wiki_graph)
    deg_cent_series = pd.Series(deg_cent, name='Degree_Centrality')
    print(deg_cent_series)
    print(deg_cent_series.sort_values(ascending=False))

    # Plot a histogram of the degree distribution of the graph
    # Compute the degree (n neighbours) of every node: degrees
    degrees = [len(list(wiki_graph.neighbors(n))) for n in wiki_graph.nodes()]
    degrees_series = pd.Series(degrees, name='Degrees')
    print(degrees_series)
    print(degrees_series.sort_values(ascending=False))

    # ======================================================================
    # Betweeness centrality
    # ======================================================================
    # Compute the betweenness centrality of T: bet_cen
    bet_cen = nx.betweenness_centrality(wiki_graph)
    bet_cen_series = pd.Series(bet_cen, name='Betweeness_Centrality')
    print(bet_cen_series.sort_values(ascending=False))

    # Plot a scatter plot of the centrality distribution and the degree
    # distribution
    plt.figure()
    plt.scatter(degrees, list(deg_cent.values()))
    plt.show()

    # Find the nodes that can broadcast messages very efficiently to lots of
    # people one degree of separation away.
    # Define find_nodes_with_highest_deg_cent()
    def find_nodes_with_highest_deg_cent(G):
        # Compute the degree centrality of G: deg_cent
        deg_cent = nx.degree_centrality(G)
        # Compute the maximum degree centrality: max_dc
        max_dc = max(list(deg_cent.values()))
        nodes = set()
        # Iterate over the degree centrality dictionary
        for k, v in deg_cent.items():
            # Check if the current value has the maximum degree centrality
            if v == max_dc:
                # Add the current node to the set of nodes
                nodes.add(k)
        return nodes

    # Find the node(s) that has the highest degree centrality in T: top_dc
    top_dc = find_nodes_with_highest_deg_cent(wiki_graph)
    print(top_dc)
    # Write the assertion statement
    for node in top_dc:
        assert (nx.degree_centrality(
            wiki_graph)[node] == max(nx.degree_centrality(wiki_graph)
                                     .values())
                )

    # Now for betweeness centrality
    # Define find_node_with_highest_bet_cent()
    def find_node_with_highest_bet_cent(G):
        # Compute betweenness centrality: bet_cent
        bet_cent = nx.betweenness_centrality(G)
        # Compute maximum betweenness centrality: max_bc
        max_bc = max(list(bet_cent.values()))
        nodes = set()
        # Iterate over the betweenness centrality dictionary
        for k, v in bet_cent.items():
            # Check if the current value has the maximum betweenness
            # centrality
            if v == max_bc:
                # Add the current node to the set of nodes
                nodes.add(k)
        return nodes

    # Use that function to find the node(s) that has the highest betweenness
    # centrality in the network: top_bc
    top_bc = find_node_with_highest_bet_cent(wiki_graph)
    print(top_bc)

    # Write an assertion statement that checks that the node(s) is/are
    # correctly identified.
    for node in top_bc:
        assert nx.betweenness_centrality(wiki_graph)[node] == max(
            nx.betweenness_centrality(wiki_graph).values()
        )

    # ======================================================================
    # Cliques
    # ======================================================================
    # Identifying triangle relationships (the simplest complex clique)
    from itertools import combinations

    # Write a function that identifies all nodes in a triangle relationship
    # with a given node.
    def nodes_in_triangle(G, n):
        """
        Returns the nodes in a graph `G` that are involved in a triangle
        relationship with the node `n`.
        """
        triangle_nodes = set([n])
        # Iterate over all possible triangle relationship combinations
        for n1, n2 in combinations(G.neighbors(n), 2):
            # Check if n1 and n2 have an edge between them
            if G.has_edge(n1, n2):
                # Add n1 to triangle_nodes
                triangle_nodes.add(n1)
                # Add n2 to triangle_nodes
                triangle_nodes.add(n2)
        return triangle_nodes

    # Print and write the assertion statement
    print(nodes_in_triangle(wiki_graph, 'Andreas_Leigh_Aabel'))
    assert len(nodes_in_triangle(wiki_graph, 'Andreas_Leigh_Aabel')) == 1

    # Fnding open triangles. form the basis of friend recommendation systems;
    # if "A" knows "B" and "A" knows "C", then it's probable that "B" also
    # knows "C".
    # Define node_in_open_triangle()
    def node_in_open_triangle(G, n):
        """
        Checks whether pairs of neighbors of node `n` in graph `G` are in an
        'open triangle' relationship with node `n`.
        """
        in_open_triangle = False
        # Iterate over all possible triangle relationship combinations
        for n1, n2 in combinations(G.neighbors(n), 2):
            # Check if n1 and n2 do NOT have an edge between them
            if not G.has_edge(n1, n2):
                in_open_triangle = True
                break
        return in_open_triangle

    # Compute the number of open triangles in T
    num_open_triangles = 0
    # Iterate over all the nodes in T
    for n in wiki_graph.nodes():
        # Check if the current node is in an open triangle
        if node_in_open_triangle(wiki_graph, n):
            # Increment num_open_triangles
            num_open_triangles += 1
    print(num_open_triangles)

    # Finding maximal cliques
    # Define maximal_cliques()
    def maximal_cliques(G, size):
        """
        Finds all maximal cliques in graph `G` that are of size `size`.
        """
        mcs = []
        for clique in nx.find_cliques(G):
            if len(clique) == size:
                mcs.append(clique)
        return mcs

    print(len(maximal_cliques(wiki_graph, 2)))  # 2658928
    assert len(maximal_cliques(wiki_graph, 2)) == 2658928

    subset = parsed_df.sample(n=1000)
    subset_graph = nx.from_pandas_edgelist(df=subset,
                                           source='Source',
                                           target='Target',
                                           edge_attr=True,
                                           create_using=nx.Graph())
    print(nx.info(subset_graph))
    print(nx.draw(subset_graph))

    # ======================================================================
    # Subgraphs
    # ======================================================================
    # There may be times when you just want to analyze a subset of nodes
    # in a network. To do so, you can copy them out into another graph
    # object using G.subgraph(nodes), which returns a new graph object
    # (of the same type as the original graph) that is comprised of the
    # iterable of nodes that was passed in.
    nodes_of_interest = [29, 38, 42]  # provided.

    # Define get_nodes_and_nbrs()
    def get_nodes_and_nbrs(G, nodes_of_interest):
        """
        Returns a subgraph of the graph `G` with only the `nodes_of_interest`
        and their neighbors.
        """
        nodes_to_draw = []
        # Iterate over the nodes of interest
        for n in nodes_of_interest:
            # Append the nodes of interest to nodes_to_draw
            nodes_to_draw.append(n)
            # Iterate over all the neighbors of node n
            for nbr in G.neighbors(n):
                # Append the neighbors of n to nodes_to_draw
                nodes_to_draw.append(nbr)
        return G.subgraph(nodes_to_draw)

    # Extract the subgraph with the nodes of interest: T_draw
    T_draw = get_nodes_and_nbrs(wiki_graph, ["'Nadine_Marshall'",
                                             "'Park_Sang-myun'"])

    # Draw the subgraph to the screen
    nx.draw(T_draw)
    plt.show()

    # # Extract the nodes of interest: nodes
    # nodes = [n for n, d in wiki_graph.nodes(data=True) if d['occupation'] == 'celebrity']
    # # Create the set of nodes: nodeset
    # nodeset = set(nodes)
    #
    # # Iterate over nodes
    # for n in nodes:
    #     # Compute the neighbors of n: nbrs
    #     nbrs = T.neighbors(n)
    #     # Compute the union of nodeset and nbrs: nodeset
    #     nodeset = nodeset.union(nbrs)
    #
    # # Compute the subgraph using nodeset: T_sub
    # T_sub = T.subgraph(nodeset)
    #
    # # Draw T_sub to the screen
    # nx.draw(T_sub)
    # plt.show()

    print(nx.adjacency_matrix(subset_graph).todense())
    print(pd.DataFrame(
        nx.to_scipy_sparse_matrix(subset_graph)
            .todense())
          .to_csv('Sparse_Adjacency_Matrix.csv')
          )
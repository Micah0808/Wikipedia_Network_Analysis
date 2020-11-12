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

    pandas_config()
    raw_df = pd.read_json(data_path)
    print(raw_df.head())
    #                        name                     links                 dob
    # 0       Andreas_Leigh_Aabel  [Anders_Krogvig, Carl...  [1830, None, None]
    # 1   Benjamin_Vaughan_Abbott  [Austin_Abbott, Edwar...      [1830, jun, 4]
    # 2          Burroughs_Abbott           [James_H._Kyle]  [1830, None, None]
    # 3  Robert_Abbott_(politi...                        []     [1830, oct, 31]
    # 4            Abd%C3%BClaziz  [%C5%9Eehzade_Yusuf_I...  [1830, None, None]

    # print(pd.to_datetime(raw_df['dob']))

    print(raw_df.columns)
    # Index(['name', 'links', 'dob'], dtype='object')

    print(raw_df.isna().mean())
    # name     0.0
    # links    0.0
    # dob      0.0

    print(raw_df.dtypes)
    # name     object
    # links    object
    # dob      object

    # How many observations are missing links to others in the df?
    print(raw_df.loc[raw_df['links'].str.len() == 0].shape)  # 528340
    raw_df = raw_df.loc[raw_df['links'].str.len() != 0]  # Let's drop these
    print(raw_df.shape)  # 731070 remaining
    print(raw_df.head())
    #                       name                     links                 dob
    # 0      Andreas_Leigh_Aabel  [Anders_Krogvig, Carl...  [1830, None, None]
    # 1  Benjamin_Vaughan_Abbott  [Austin_Abbott, Edwar...      [1830, jun, 4]
    # 2         Burroughs_Abbott           [James_H._Kyle]  [1830, None, None]
    # 4           Abd%C3%BClaziz  [%C5%9Eehzade_Yusuf_I...  [1830, None, None]
    # 7            Santos_Acosta    [Joaqu%C3%ADn_Riascos]  [1830, None, None]

    # Creating a data frame of each list element in dob to inspect for NaNs
    # None values. This will give me a better idea of what we have to work
    # with.
    date_df = (pd.DataFrame(raw_df['dob']
                            .astype(str)
                            .str.split()
                            .values
                            .tolist(),
                            columns=['Year', 'Month', 'Day']))

    # How many are missing values or are None in each date column?
    print(date_df['Year'].isna().sum())  # 0 missing values
    print(date_df['Month'].value_counts()[0])  # 381705
    print(date_df['Day'].value_counts()[0])  # 381705

    # Same amount of missing values between Months and day. Is this coincidental
    # or does missing in one column mean that the other will also be missing?
    print(date_df.loc[date_df['Month'].str.contains('None')
                      & date_df['Day'].str.contains('None')]
          .shape)  # 381705

    # Yes, this is true. If a value is None in either Month or Day
    # it is also None in the other.

    # Now cleaning up this date dataframe into a usable metadata format.
    date_df = (date_df
               .apply(lambda x: x.str.strip('['))  # Stripping lead chars
               .apply(lambda x: x.str.strip(']'))  # Stripping lag chars
               .apply(lambda x: x.str.strip(','))  # Stripping lag chars
               .replace('None', np.nan)  # Prepping data for datetime format
               .assign(Month=lambda x: x.Month.replace({"'jan'": 1,
                                                        "'feb'": 2,
                                                        "'mar'": 3,
                                                        "'apr'": 4,
                                                        "'may'": 5,
                                                        "'jun'": 6,
                                                        "'jul'": 7,
                                                        "'aug'": 8,
                                                        "'sep'": 9,
                                                        "'oct'": 10,
                                                        "'nov'": 11,
                                                        "'dec'": 12}))
               .assign(Day=lambda x: x.Day.astype(float))  # Parsing dtypes
               .assign(Year=lambda x: x.Year.astype(int))  # Parsing dtypes
               .assign(Datetime=lambda x: pd.to_datetime(arg=x,  # To datetime
                                                         dayfirst=True,
                                                         errors='coerce'))
               .filter(items=['Year', 'Datetime']))  # Selecting final metadata
    print(date_df)

    # Now creating a wide form df for the link elements
    wide_link_df = (
        raw_df['links']
        .astype(str)  # Parsing to strings for data munging
        .str.split(expand=True)  # Expanding each list element to a column
        .apply(lambda x: x.str.strip('['))  # Stripping lead chars
        .apply(lambda x: x.str.strip(']'))  # Stripping lag chars
        .apply(lambda x: x.str.strip(','))  # Stripping lag chars
        .add_prefix('Link_')  # Adding a prefix to each column name
        .fillna(value=np.nan)  # Replacing None values with NaNs
    )
    print(wide_link_df)

    # Inserting Datetime, Year, Name and ID columns at the start of the df
    # to parse from a wide to long format
    wide_link_df.insert(loc=0, column='Datetime', value=date_df['Datetime'])
    wide_link_df.insert(loc=0, column='Year', value=date_df['Year'])
    wide_link_df.insert(loc=0, column='Name', value=raw_df['name'])
    wide_link_df.insert(loc=0, column='ID', value=wide_link_df.index)
    print(wide_link_df['Year'].value_counts())
    print(wide_link_df.head())

    # head_df = wide_link_df.head(20)
    # print(head_df)
    # print(pd.wide_to_long(head_df, stubnames='Link_', i='ID', j='Name'))

    # Converting from a wide to long dataframe
    long_parsed_df = (
        pd.wide_to_long(wide_link_df, stubnames='Link_', i='ID', j='Name')
        .query('Link_.notna() | Year.notna()')  # Dropping NaNs
        .rename(columns={'Name': 'Source', 'Link_': 'Target'})
        .reset_index(drop=True)
    )
    print(long_parsed_df.shape)
    print(long_parsed_df.head(10))

    #                      Source                    Target
    # 0       Andreas_Leigh_Aabel          'Anders_Krogvig'
    # 1   Benjamin_Vaughan_Abbott           'Austin_Abbott'
    # 2          Burroughs_Abbott           'James_H._Kyle'
    # 3            Abd%C3%BClaziz  '%C5%9Eehzade_Yusuf_I...
    # 4             Santos_Acosta    'Joaqu%C3%ADn_Riascos'
    # 5          John_Adams-Acton  'Anthony_Ashley-Coope...
    # 6          John_Hicks_Adams       'Benjamin_Mayfield'
    # 7  Walter_Adams_(Austral...      'George_Joseph_Hall'
    # 8               Aga_Khan_II            'Aga_Khan_III'
    # 9          Karin_%C3%85hlin     'Anna_Sandstr%C3%B6m'

    long_parsed_df.to_csv('long_parsed_df_with_meta_data.csv')
    long_parsed_df.to_pickle('long_parsed_df_with_meta_data.pkl')

    # =========================================================================
    # Testing out a NetworkX object
    # =========================================================================
    # Reading in, inspecting, and plotting the graph
    parsed_df = pd.read_pickle('long_parsed_df_with_meta_data.pkl')
    parsed_df = parsed_df.query('Year.notna()')
    print(parsed_df.shape)  # (2658928, 4)

    wiki_graph = nx.from_pandas_edgelist(df=parsed_df,
                                         source='Source',
                                         target='Target',
                                         edge_attr='Year',
                                         create_using=nx.Graph())
    print(type(wiki_graph))
    print(wiki_graph.nodes())
    print(wiki_graph.edges())
    print(nx.info(wiki_graph))
    nx.draw(wiki_graph)

    # Type: Graph
    # Number of nodes: 950115
    # Number of edges: 2658928
    # Average degree:   5.5971

    # ======================================================================
    # Visualising
    # ======================================================================
    # # Create rationale plots
    # In an undirected graph, the matrix is symmetrical around the diagonal
    # as in this plot. Therefore, the data is read in correctly.
    # m = nv.MatrixPlot(wiki_graph)
    # m.draw()
    # plt.show()
    # c = nv.CircosPlot(wiki_graph)
    # c.draw()
    # plt.show()
    # a = nv.ArcPlot(wiki_graph)
    # a.draw()
    # plt.show()

    # ======================================================================
    # Testing a model
    # ======================================================================
    from networkx.algorithms.community.centrality import girvan_newman
    import itertools

    G = nx.path_graph(10)
    print(G.nodes())  # NodeView((0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
    print(G.edges())
    # EdgeView([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
    # (5, 6), (6, 7), (7, 8), (8, 9)])

    # Model
    # To get the first pair of communities:
    comp = girvan_newman(wiki_graph)
    print(tuple(sorted(c) for c in next(comp)))

    # To get only the first k tuples of communities, use itertools.islice():
    k = 10
    comp = girvan_newman(wiki_graph)
    for communities in itertools.islice(comp, k):
        print(tuple(sorted(c) for c in communities))

    # To stop getting tuples of communities once the number of communities is
    # greater than k, use itertools.takewhile():
    G = nx.path_graph(8)
    k = 500
    comp = girvan_newman(wiki_graph)
    limited = itertools.takewhile(lambda c: len(c) <= k, comp)
    for communities in limited:
        print(tuple(sorted(c) for c in communities))

    # ======================================================================
    # Degree centrality
    # ======================================================================
    # Degree centrality. Who has the most neighbours compared to all possible
    # neighbours in the dataset?
    deg_cent = nx.degree_centrality(wiki_graph)

    # Plot a histogram of the degree centrality distribution of the graph
    plt.figure()
    plt.hist(list(deg_cent.values()), bins=100, range=[0.0000, 0.00005])
    plt.show()

    # Plot a histogram of the degree distribution of the graph
    # Compute the degree (n neighbours) of every node: degrees
    degrees = [len(list(wiki_graph.neighbors(n))) for n in wiki_graph.nodes()]
    plt.figure()
    plt.hist(degrees, bins=100, range=[min(degrees), 75])
    plt.show()

    # Plot a scatter plot of the centrality distribution and the degree
    # distribution
    plt.figure()
    plt.scatter(degrees, list(deg_cent.values()))
    plt.show()


    def path_exists(G, node1, node2):
        """
        This function checks whether a path exists between two nodes
        (node1, node2) in graph G.
        """
        visited_nodes = set()
        queue = [node1]

        for node in queue:
            neighbors = list(G.neighbors(node))
            if node2 in neighbors:
                print('Path exists between nodes {0} and {1}'.format(node1,
                                                                     node2))
                return True
                break
            else:
                visited_nodes.add(node)
                queue.extend([n for n in neighbors if n not in visited_nodes])
            # Check to see if the final element of the queue has been reached
            if node == queue[-1]:
                print('Path does not exist between nodes {0} and {1}'.format(
                    node1, node2))
                # Place the appropriate return statement
                return False

    print(path_exists(wiki_graph, "'Richard_F._Thomas'", "'Jennie_Stoller'"))
    print(pd.Series(deg_cent, name='Degree_Centrality').sort_values())

    def nodes_with_m_nbrs(G, m):
        """
        Returns all nodes in graph G that have m neighbors.
        """
        nodes = set()
        # Iterate over all nodes in G
        for n in G.nodes():
            # Check if the number of neighbors of n matches m
            if len(list(G.neighbors(n))) == m:
                # Add the node n to the set
                nodes.add(n)
        # Return the nodes with m neighbors
        return nodes

    print(nodes_with_m_nbrs(wiki_graph, 10))

    # Compute the degree (n neighbours) of every node: degrees
    degrees = [len(list(wiki_graph.neighbors(n))) for n in wiki_graph.nodes()]
    print(degrees)

    # ======================================================================
    # Betweeness centrality
    # ======================================================================
    # Compute the betweenness centrality of T: bet_cen
    bet_cen = nx.betweenness_centrality(wiki_graph)
    print(pd.Series(bet_cen, name='Betweeness_Centrality').sort_values())
    deg_cen = nx.degree_centrality(wiki_graph)  # Compute the degree
    print(pd.Series(deg_cen, name='Degree_Centrality').sort_values())
    # Create a scatter plot of betweenness centrality and degree centrality
    plt.scatter(list(bet_cen.values()), list(deg_cen.values()))
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











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
        .query('Link_.notna()')  # Dropping NaNs
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
    pandas_config()
    parsed_df = pd.read_pickle('long_parsed_df_with_meta_data.pkl')
    parsed_df = parsed_df.drop(['Datetime', 'Year'], axis=1)
    parsed_df['Target'] = parsed_df['Target'].str.strip("'")
    print(parsed_df.head())

    print(parsed_df.columns)
    print(parsed_df.shape)  # (4353922, 2)
    print(parsed_df.dtypes)
    # Source    object
    # Target    object

    # Creating a networkx graph
    sub = parsed_df.sample(1000, random_state=10)
    wiki_graph = nx.from_pandas_edgelist(df=sub,
                                         source='Source',
                                         target='Target',
                                         create_using=nx.Graph())
    print(type(wiki_graph))
    print(wiki_graph.nodes(data=True))
    print(wiki_graph.edges())
    print(nx.info(wiki_graph))
    nx.draw(wiki_graph)

    # Type: Graph
    # Number of nodes: 841055
    # Number of edges: 3695365
    # Average degree:   8.7875

    # # Is there a relationship between year and degree
    # degrees = [len(list(wiki_graph.neighbors(n))) for n in wiki_graph.nodes()]
    # print(len(degrees))
    #
    # corr_df = parsed_df.assign(Degree=degrees)
    #
    # print(parsed_df.groupby('Source').count().shape)
    # print(parsed_df.loc[parsed_df['Source'].duplicated(keep='first') == False].shape)


    # ======================================================================
    # Testing a model
    # ======================================================================
    from networkx.algorithms.community.centrality import girvan_newman
    import itertools

    # G = nx.path_graph(10)

    # # Creating a networkx graph
    # subsample = parsed_df.sample(n=10000, random_state=10)
    # # subsample = subsample['Source'].str.strip('"')
    #
    # subsample['Source_Int'] = pd.factorize(subsample['Source'])[0]
    # subsample['Source'].value_counts()
    # subsample['Source_Int'].value_counts()
    #
    # subsample['Target_Int'] = pd.factorize(subsample['Target'])[0]
    # subsample['Target'].value_counts()
    # subsample['Target_Int'].value_counts()
    #
    # subsample = subsample[['Source_Int', 'Target_Int']]
    # print(subsample.dtypes)
    # print(subsample)
    # print(subsample.shape)
    #
    # G = nx.from_pandas_edgelist(df=subsample,
    #                             source='Source_Int',
    #                             target='Target_Int',
    #                             create_using=nx.Graph())
    # print(nx.info(G))
    # Number of nodes: 9748
    # Number of edges: 10000
    # Average degree:   2.0517

    from node2vec import Node2Vec
    # Generate walks
    node2vec = Node2Vec(wiki_graph, dimensions=2, walk_length=20, num_walks=10, workers=4)
    model = node2vec.fit(window=10, min_count=1)  # Learn embeddings
    # Save the embedding in file embedding.emb
    model.wv.save_word2vec_format("embedding.emb")

    # Clustering model
    X = np.loadtxt("embedding.emb", dtype=str, skiprows=1)
    print(pd.DataFrame(X))
    X = X[X[:, 0].argsort()];
    print(X)
    print(pd.DataFrame(X))

    # Remove the node index from X and save in Z
    Z = X[0:X.shape[0], 1:X.shape[1]];
    print(pd.DataFrame(Z))

    from sklearn.cluster import AgglomerativeClustering
    clust = AgglomerativeClustering(n_clusters=2).fit(Z)
    labels = clust.labels_  # get the cluster labels of the nodes.
    print(len(labels))
    print(labels)
    print(pd.Series(labels).value_counts())

    from sklearn.metrics import silhouette_score
    print(silhouette_score(pd.DataFrame(X)[0].factorize()[0].reshape(-1, 1),
                           labels))













    # # To stop getting tuples of communities once the number of communities is
    # # greater than k, use itertools.takewhile():
    # G = nx.path_graph(8)
    # k = 500
    # comp = girvan_newman(wiki_graph)
    # limited = itertools.takewhile(lambda c: len(c) <= k, comp)
    # for communities in limited:
    #     print(tuple(sorted(c) for c in communities))

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













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
import networkx as nx
import nxviz as nv
from node2vec import Node2Vec
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from networkx.algorithms.community.centrality import girvan_newman
import matplotlib.pyplot as plt
import itertools

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
    # Reading in, inspecting, and plotting the graph
    pandas_config()
    parsed_df = pd.read_pickle('long_parsed_df_with_meta_data.pkl')
    parsed_df = parsed_df.drop(['Datetime', 'Year'], axis=1)
    parsed_df['Target'] = parsed_df['Target'].str.strip("'")  # Final clean
    print(parsed_df.head())

    print(parsed_df.columns)
    print(parsed_df.shape)  # (4353922, 2)
    print(parsed_df.dtypes)
    # Source    object
    # Target    object

    # Creating a networkx graph
    sub = parsed_df.sample(1000, random_state=10)  # Taking a sub-sample for now
    wiki_graph = nx.from_pandas_edgelist(df=parsed_df,
                                         source='Source',
                                         target='Target',
                                         create_using=nx.Graph())
    print(type(wiki_graph))
    print(wiki_graph.nodes(data=True))
    print(wiki_graph.edges())
    print(nx.info(wiki_graph))
    # nx.draw(wiki_graph)

    # Type: Graph
    # Number of nodes: 841055
    # Number of edges: 3695365
    # Average degree:   8.7875

    # ======================================================================
    # Visualising
    # ======================================================================

    # PLOT AGE AND DEGREE DISTRIBUTION

    # Create rationale plots
    # In an undirected graph, the matrix is symmetrical around the diagonal
    # as in this plot. Therefore, the data is read in correctly.
    m = nv.MatrixPlot(wiki_graph)
    m.draw()
    plt.show()
    c = nv.CircosPlot(wiki_graph)
    c.draw()
    plt.show()
    a = nv.ArcPlot(wiki_graph)
    a.draw()
    plt.show()

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




    # plt.figure()
    # plt.hist(degrees, bins=100, range=[min(degrees), 75])
    # plt.show()

    # Plot a scatter plot of the centrality distribution and the degree
    # distribution
    plt.figure()
    plt.scatter(degrees, list(deg_cent.values()))
    plt.show()

    # ======================================================================
    # Girvan Newman Method
    # ======================================================================
    # To get only the first k tuples of communities, use itertools.islice():
    k = 3
    comp = girvan_newman(wiki_graph)
    for communities in itertools.islice(comp, k):
        print(tuple(sorted(c) for c in communities))

    # ======================================================================
    # Clustering
    # ======================================================================
    # Generating walks
    node2vec = Node2Vec(wiki_graph, dimensions=2, walk_length=20, num_walks=10,
                        workers=4)
    model = node2vec.fit(window=10, min_count=1)  # Learn embeddings
    model.wv.save_word2vec_format("embedding.emb")

    # Data prep
    X = np.loadtxt("embedding.emb", dtype=str, skiprows=1)
    print(pd.DataFrame(X))

    X = X[X[:, 0].argsort()];
    print(X)
    print(pd.DataFrame(X))

    # Remove the node index from X and save in Z
    Z = X[0:X.shape[0], 1:X.shape[1]];
    print(pd.DataFrame(Z))

    # Clustering
    clust = AgglomerativeClustering(n_clusters=2).fit(Z)
    labels = clust.labels_  # get the cluster labels of the nodes.
    print(len(labels))
    print(labels)
    print(pd.Series(labels).value_counts())

    # Checking cluster score
    numeric_X = pd.DataFrame(X)[0].factorize()[0].reshape(-1, 1)
    print(silhouette_score(numeric_X, labels))


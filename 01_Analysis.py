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
from scipy.sparse.csgraph import connected_components

import sknetwork as skn
from sknetwork.clustering import (modularity, PropagationClustering)
from sknetwork.data import load_graphml
import time

data_path = ('/Users/MicahJackson/anaconda/Pycharm_WD/'
             'Wikipedia_Network_Analysis/Data/Wikipedia_Raw.json')

output_dir = ('/Users/MicahJackson/anaconda/Pycharm_WD/'
              'Wikipedia_Network_Analysis/Data')

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

    # =========================================================================
    parsed_df = parsed_df.sample(100, random_state=10)
    wiki_graph = nx.from_pandas_edgelist(df=parsed_df,
                                         source='Source',
                                         target='Target',
                                         create_using=nx.Graph())

    print(type(wiki_graph))
    print(pd.Series(wiki_graph.nodes()).shape)
    print(wiki_graph.edges())
    print(nx.info(wiki_graph))
    nx.write_graphml(wiki_graph, path='wiki_graph.graphml')

    # Type: Graph
    # Number of nodes: 841055
    # Number of edges: 3695365
    # Average degree:   8.7875

    graph = load_graphml('wiki_graph.graphml')
    adjacency = graph.adjacency
    names = graph.names
    print(adjacency)
    print(names)

    adjacency_array = adjacency.toarray().astype(int)
    print(adjacency_array)
    # [[0 1 0 ... 0 0 0]
    #  [1 0 0 ... 0 0 0]
    #  [0 0 0 ... 0 0 0]
    #  ...
    #  [0 0 0 ... 0 0 0]
    #  [0 0 0 ... 0 0 1]
    #  [0 0 0 ... 0 1 0]]

    print(adjacency_array.transpose())
    # [[0 1 0 ... 0 0 0]
    #  [1 0 0 ... 0 0 0]
    #  [0 0 0 ... 0 0 0]
    #  ...
    #  [0 0 0 ... 0 0 0]
    #  [0 0 0 ... 0 0 1]
    #  [0 0 0 ... 0 1 0]]

    # =========================================================================
    # Running the clustering
    # =========================================================================
    t0 = time.time()
    prop = PropagationClustering()
    labels = prop.fit_transform(adjacency)
    t1 = time.time()
    total = t1 - t0
    print(total)
    # 23.39259171485901

    unique_labels, counts = np.unique(labels, return_counts=True)
    print(unique_labels, counts)

    # image = svg_graph(adjacency, labels=labels)
    # svg_graph(adjacency, names, labels=labels, filename='wiki_prop_labels')
    modularity(adjacency, labels)

    node_series = pd.Series(wiki_graph.nodes(), name='Nodes')
    clust_labels = pd.Series(labels, name='Cluster')
    nodes_labels = pd.concat([node_series, clust_labels], axis=1)
    print(node_series)
    print(clust_labels)
    print(nodes_labels)

    final_df = (pd.merge(left=nodes_labels,
                         right=parsed_df,
                         left_on='Nodes',
                         right_on='Source')
                .drop('Nodes', axis=1))
    print(final_df)

    # print(final_df.loc[final_df['Cluster'].value_counts() <= 20])

    # Filtering down to smaller circles. This is because we want to introduce
    # as many people to circles as possible without creating circles that
    # feel wierd.
    # print(final_df.loc[final_df['Source'].map(final_df['Source'])])
    print(final_df['Source'].value_counts())

    # Filtering out duplicates in the long dataframe to avoid people ending up
    # being counted multiple times in a cluster
    final_df = (final_df
                .loc[final_df['Source']
                              .duplicated(keep='first') == False])

    filt_df = (final_df
               .loc[final_df['Cluster'].map(final_df['Cluster']
                                            .value_counts()) <= 46])
    filt_df = (filt_df
               .loc[final_df['Cluster'].map(final_df['Cluster']
                                            .value_counts()) >= 2])

    print(filt_df.shape)  # 27676 individuals
    print(filt_df['Cluster'].value_counts().shape)  # In 10,000 circles

    # What do the people in each cluster look like? Let's look at cluster 5
    # and get rid of the duplicated names in long format.
    print(filt_df['Cluster'].unique())
    print(filt_df['Cluster'].value_counts())
    print(filt_df.loc[filt_df['Cluster'] == 5])

    # We can see that they're all Russian or have an interest in Russian
    # history

    #          Cluster              Source              Target
    # 225119         5     Nikolai_Chekhov       Anton_Chekhov
    # 1221390        5   Galina_Konovalova       Anton_Chekhov
    # 1870688        5  Marietta_Chudakova  Alexander_Chudakov
    # 1870690        5  Alexander_Chudakov       Anton_Chekhov
    # 2127515        5       Staffan_Skott       Anton_Chekhov
    # 4139625        5      Andrey_Borisov     Nikolai_Chekhov

    # What about a different circle?
    print(filt_df.loc[filt_df['Cluster'] == 20035])

    # All japanese kick boxers
    #          Cluster            Source           Target
    # 4347387    20035     Ryuya_Okuwaki  Haruto_Yasumoto
    # 4347389    20035   Haruto_Yasumoto    Ryuya_Okuwaki
    # 4350524    20035  Nadaka_Yoshinari      Issei_Ishii

    # What about a different circle?
    print(filt_df.loc[filt_df['Cluster'] == 20147])

    # Twin sisters who both play football. One's a defender, the other is a
    # forward.

    #          Cluster           Source           Target
    # 4353634    20147      Anai_Mogini  Anuching_Mogini
    # 4353635    20147  Anuching_Mogini      Anai_Mogini

    # What about our two biggest circles?
    print(filt_df.loc[filt_df['Cluster'] == 3])

    #          Cluster                    Source                    Target
    # 81             3  Walter_Adams_(Austral...        George_Joseph_Hall
    # 83             3        George_Joseph_Hall  Michael_Duffy_(Queens...
    # 11796          3    Lewis_Adolphus_Bernays      James_Laurence_Watts
    # 15186          3  George_Harris_(Queens...       Charles_Lumley_Hill
    # 15191          3       Charles_Lumley_Hill  Frederick_Cooper_(pol...
    #           ...                       ...                       ...
    # 251160         3     Arthur_Charles_Cooper  Charles_Borromeo_Fitz...
    # 254334         3        Arthur_Whittingham       Charles_Lumley_Hill
    # 339008         3               Richard_Bow  John_Payne_(Queenslan...
    # 760915         3          Paddy_Fitzgerald  Charles_Borromeo_Fitz...
    # 1302416        3           Beaufort_Palmer               Hugh_Mosman

    # All appear to be Australians from Queensland in the 1800s

    print(filt_df.loc[filt_df['Cluster'] == 9645])
    #          Cluster                    Source                    Target
    # 2340955     9645        Daniel_Levavasseur     Laura_Flessel-Colovic
    # 2587605     9645           Octavian_Zidaru  Amalia_T%C4%83t%C4%83ran
    # 2587618     9645  Amalia_T%C4%83t%C4%83ran          Corinna_Lawrence
    # 3220813     9645                Adrian_Pop  Bj%C3%B6rne_V%C3%A4gg...
    # 3345393     9645              Cornel_Milan          Anca_M%C4%83roiu
    #           ...                       ...                       ...
    # 4208004     9645           Julia_Beljajeva              Anna_Sivkova
    # 4232165     9645                 Sun_Yujie        Daniel_Levavasseur
    # 4235130     9645                   Xu_Anqi        Daniel_Levavasseur
    # 4255759     9645         Andrea_Santarelli            Yuval_Freilich
    # 4255760     9645            Yuval_Freilich                Noam_Mills

    # ======================================================================
    # Visualising
    # ======================================================================
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

    # # ======================================================================
    # # Betweeness centrality
    # # ======================================================================
    # # Compute the betweenness centrality of T: bet_cen
    # bet_cen = nx.betweenness_centrality(wiki_graph)
    # bet_cen_series = pd.Series(bet_cen, name='Betweeness_Centrality')
    # print(bet_cen_series.sort_values(ascending=False))
    #
    # # Plot a scatter plot of the centrality distribution and the degree
    # # distribution
    # plt.figure()
    # plt.scatter(degrees, list(deg_cent.values()))
    # plt.show()

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
    # 100 embeddings recomended by Sina to start with
    node2vec = Node2Vec(wiki_graph, dimensions=100, walk_length=20, num_walks=10,
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


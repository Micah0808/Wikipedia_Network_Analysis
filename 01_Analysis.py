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
from sklearn.metrics import silhouette_score
from scipy.sparse.csgraph import connected_components
from scipy.sparse.csgraph import csgraph_to_dense

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
    # Undirected
    # =========================================================================
    undir_wiki_graph = nx.from_pandas_edgelist(df=parsed_df,
                                               source='Source',
                                               target='Target',
                                               create_using=nx.Graph())

    print(type(undir_wiki_graph))
    print(pd.Series(undir_wiki_graph.nodes()).shape)
    print(undir_wiki_graph.edges())
    print(nx.info(undir_wiki_graph))

    # Writing this to a undirected graphml file for the undirected analysis in
    # task 1
    nx.write_graphml(undir_wiki_graph, path='undir_wiki_graph.graphml')

    # Undirected
    # Number of nodes: 841055
    # Number of edges: 3695365
    # Average degree:   8.7875

    # Getting all the edges in the undirected graph and parsing to columns in a
    # pandas dataframe
    undir_edges = undir_wiki_graph.edges()
    undir_edges_series = pd.Series(undir_edges, name='Edges')
    undir_edges_df = pd.DataFrame(undir_edges_series
                                  .values
                                  .tolist(),
                                  index=undir_edges_series.index,
                                  columns=['Edge_1', 'Edge_2'])
    print(undir_edges_df.dtypes)
    print(undir_edges_df.shape)  # 3695365 as expected
    print(undir_edges_df)

    # =========================================================================
    # Directed
    # =========================================================================
    dir_wiki_graph = nx.from_pandas_edgelist(df=parsed_df,
                                             source='Source',
                                             target='Target',
                                             create_using=nx.DiGraph())

    print(type(dir_wiki_graph))  # networkx.classes.digraph.DiGraph
    print(nx.info(dir_wiki_graph))

    # Directed
    # Type: DiGraph
    # Number of nodes: 841055
    # Number of edges: 4353922
    # Average in degree:   5.1767
    # Average out degree:   5.1767

    # Getting all the edges in the directed graph and parsing to columns in a
    # pandas dataframe
    dir_edges = dir_wiki_graph.edges()
    dir_edges_series = pd.Series(dir_edges, name='Edges')
    dir_edges_df = pd.DataFrame(dir_edges_series
                                .values
                                .tolist(),
                                index=dir_edges_series.index,
                                columns=['Edge_1', 'Edge_2'])
    print(dir_edges_df.dtypes)
    print(dir_edges_df.shape)  # 4353922 as expected
    print(dir_edges_df)

    # Finding those in the directed graph edge list who are not in the
    # undirected graph edge list. First looking in edge 1.
    directed_df_edge_one = (
        dir_edges_df.loc[dir_edges_df['Edge_1']
                         .isin(undir_edges_df['Edge_1']) == False]
    )
    print(directed_df_edge_one.shape)  # (447902, 2)
    print(directed_df_edge_one['Edge_1'].value_counts())
    print(directed_df_edge_one)

    # Now edge 2
    directed_df_edge_two = (
        dir_edges_df.loc[dir_edges_df['Edge_2']
                         .isin(undir_edges_df['Edge_2']) == False]
    )
    print(directed_df_edge_two.shape)  # (75595, 2)
    print(directed_df_edge_two['Edge_2'].value_counts())
    print(directed_df_edge_two)

    # Concatenating the two together
    final_dir_edge_df = pd.concat([directed_df_edge_one, directed_df_edge_two])
    final_dir_edge_df = final_dir_edge_df.rename(columns={'Edge_1': 'Source',
                                                          'Edge_2': 'Target'})
    print(final_dir_edge_df)
    print(final_dir_edge_df.shape)  # (523497, 2)

    # =========================================================================

    # Now setting up a networkx graph object only using the final directed
    # edge subset
    directed_wiki_graph = nx.from_pandas_edgelist(df=final_dir_edge_df,
                                                  source='Source',
                                                  target='Target',
                                                  create_using=nx.DiGraph())
    print(type(directed_wiki_graph))  # networkx.classes.digraph.DiGraph
    print(nx.info(directed_wiki_graph))

    # Type: DiGraph
    # Number of nodes: 451386
    # Number of edges: 501784
    # Average in degree:   1.1117
    # Average out degree:   1.1117

    # Writing this to a directed graphml file for analysis
    nx.write_graphml(directed_wiki_graph, path='directed_wiki_graph.graphml')

    # # Using a matrix plot to
    # m = nv.MatrixPlot(directed_wiki_graph)
    # m.draw()
    # plt.show()

    # =========================================================================
    # Running the clustering for question one on the undirected graph
    # =========================================================================
    undir_graph = load_graphml('undir_wiki_graph.graphml')
    undir_adjacency = undir_graph.adjacency
    undir_names = undir_graph.names
    print(undir_adjacency)
    print(undir_adjacency.shape)  # (841055, 841055)
    print(undir_names)

    # Running the clustering model
    t0 = time.time()
    prop = PropagationClustering()
    labels = prop.fit_transform(undir_adjacency)
    t1 = time.time()
    total = t1 - t0
    print(total)
    # 23.39259171485901

    unique_labels, counts = np.unique(labels, return_counts=True)
    print(unique_labels, counts)

    # image = svg_graph(adjacency, labels=labels)
    # svg_graph(adjacency, names, labels=labels, filename='wiki_prop_labels')
    modularity(undir_adjacency, labels)  # 0.041381350642334946

    node_series = pd.Series(undir_wiki_graph.nodes(), name='Nodes')
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
    print(final_df.shape)

    # Filtering down to smaller circles. This is because we want to introduce
    # as many people to circles as possible without creating circles that
    # feel wierd.
    # print(final_df.loc[final_df['Source'].map(final_df['Source'])])
    print(final_df['Source'].value_counts())

    # Filtering out duplicates in the long dataframe to avoid people ending up
    # being counted multiple times in a cluster

    full_long_df = final_df  # But keeping full df to sample of 10,000 clusters
    final_df = (final_df
                .loc[final_df['Source']
                     .duplicated(keep='first') == False])

    # ========================================================================

    # Randomly sample different combinations of 10,000 different clusters
    # cluster N's are in the index
    seed_n_list = []
    n_users_list = []
    n_circles_list = []
    dfs_list = []
    silhouette_score_list = []
    n_samples = range(1, 10, 1)
    for n in n_samples:
        ten_k_clusters = (pd.Series(final_df['Cluster']
                                    .value_counts()  # N users in a cluster
                                    .index)
                          .sample(10000, random_state=n)  # Iterate over this
                          .tolist())

        sub_df = final_df.loc[final_df['Cluster'].isin(ten_k_clusters)]

        dfs_list.append(sub_df)
        n_users_list.append(sub_df.shape[0])
        n_circles_list.append(sub_df['Cluster'].value_counts().shape[0])
        seed_n_list.append(n)

    print(seed_n_list)
    print(n_users_list)
    print(n_circles_list)
    print(dfs_list)
    print(dfs_list[0].shape)  # (125142, 3)

    list_of_lists = [seed_n_list, n_users_list, n_circles_list]
    col_names = ['Seeds', 'N_Users', 'N_Circles']
    print(pd.DataFrame(list_of_lists, index=col_names)
          .transpose()
          .sort_values(by='N_Users', ascending=False))

    # For now, let's just take the 10,000 circles with the max number of users
    # This corresponds to a seed = 2
    clusters = (pd.Series(final_df['Cluster']
                          .value_counts()
                          .index)
                .sample(10000, random_state=2)
                .tolist())
    filt_df = final_df.loc[final_df['Cluster'].isin(clusters)]
    print(filt_df.shape)  # (22646, 3)
    print(filt_df)

    # ========================================================================

    # Random search approach
    seed = np.random.seed(10)
    min_dist = np.random.randint(low=2, high=150, size=125)
    max_dist = np.random.randint(low=2, high=150, size=125)
    print(min_dist)
    print(max_dist)

    min_list = []
    max_list = []
    n_users_list = []
    n_circles_list = []

    for min_n, max_n in zip(min_dist, max_dist):
        filt_df = (
            final_df
            .loc[final_df['Cluster'].map(final_df['Cluster']
                                         .value_counts()) <= max_n]
        )
        # Setting the minimum number of people allowed in a circle.
        filt_df = (
            filt_df
            .loc[final_df['Cluster'].map(final_df['Cluster']
                                         .value_counts()) >= min_n]
        )

        users_in_circle = filt_df.shape[0]
        number_of_circles = filt_df['Cluster'].value_counts().shape[0]

        min_list.append(min_n)
        max_list.append(max_n)
        n_users_list.append(users_in_circle)
        n_circles_list.append(number_of_circles)

    print(min_list)
    print(max_list)
    print(n_users_list)
    print(n_circles_list)

    list_of_lists = [min_list, max_list, n_users_list, n_circles_list]
    var_names = ['Min_Count', 'Max_Count', 'Users_Count', 'Circles_Count']

    print(pd.DataFrame(list_of_lists, index=var_names)
          .transpose()
          .query('Circles_Count >= 1000')
          .sort_values(by='Circles_Count'))

    #     Min_Count  Max_Count  Users_Count  Circles_Count
    # 44         17        136        32564           1000
    # 90         16         56        26619           1026
    # 70         13         23        17980           1088
    # 24         15        117        34737           1234
    # 72         14         60        31696           1335
    # 0          11         73        43102           2171
    # 57          6        114        83456           7320
    # 84          5         37        88443          10710
    # 9           2         30       321185         111173

    # Filtering based off of the searched max and min values
    filt_df = (
        final_df
        .loc[final_df['Cluster'].map(final_df['Cluster']
                                     .value_counts()) <= 136]
    )
    # Setting the minimum number of people allowed in a circle.
    filt_df = (
        filt_df
        .loc[final_df['Cluster'].map(final_df['Cluster']
                                     .value_counts()) >= 17]
    )

    # What do the people in each cluster look like?
    print(filt_df['Cluster'].unique())  # Getting the individual cluster ints
    print(filt_df['Cluster'].value_counts())  # Counting n user in each circle
    print(filt_df.loc[filt_df['Cluster'] == 84359])  # Checking circle 5

    # NOW NEED TO RE-INSPECT THESE VALUES

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

    # What about our biggest circle?
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

    # All appear to be Australians from Queensland in the 1800s.

    # =========================================================================
    # Now running the clustering for question two on the directed graph
    # =========================================================================
    dir_graph = load_graphml('directed_wiki_graph.graphml')
    dir_adjacency = dir_graph.adjacency
    dir_names = dir_graph.names
    print(dir_adjacency)
    print(dir_adjacency.shape)  # (451386, 451386)
    print(dir_names)

    # Fitting the propagation clustering
    t0 = time.time()
    prop = PropagationClustering()
    labels = prop.fit_transform(dir_adjacency)
    t1 = time.time()
    total = t1 - t0
    print(total)

    unique_labels, counts = np.unique(labels, return_counts=True)
    print(unique_labels, counts)

    # image = svg_graph(adjacency, labels=labels)
    # svg_graph(adjacency, names, labels=labels, filename='wiki_prop_labels')
    modularity(dir_adjacency, labels)  # 0.4943768068371678

    # Bringing together the clusters and nodes
    node_series = pd.Series(directed_wiki_graph.nodes(), name='Nodes')
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
    final_df.to_csv('undirected_clust_df_test.csv')

    # Filtering down to smaller circles for task 2
    print(final_df['Source'].value_counts())

    # Filtering out duplicates in the long dataframe to avoid people ending up
    # being counted multiple times in a cluster
    final_df = (final_df
                .loc[final_df['Source']
                     .duplicated(keep='first') == False])
    print(final_df.shape)

    # =========================================================================

    # Searching to find ~ 1000 circles with the max number of people
    min_range = list(range(1, 50, 1))
    max_range = list(range(50, 100, 1))[::-1]  # Reversing the list
    print(min_range)
    print(max_range)

    # =========================================================================

    # Random search approach
    seed = np.random.seed(10)
    min_dist = np.random.randint(low=2, high=150, size=125)
    max_dist = np.random.randint(low=2, high=150, size=125)
    print(min_dist)
    print(max_dist)

    min_list = []
    max_list = []
    n_users_list = []
    n_circles_list = []

    for min_n, max_n in zip(min_dist, max_dist):
        filt_df = (final_df
                   .loc[final_df['Cluster'].map(final_df['Cluster']
                                                .value_counts()) <= max_n])
        # Setting the minimum number of people allowed in a circle.
        filt_df = (filt_df
                   .loc[final_df['Cluster'].map(final_df['Cluster']
                                                .value_counts()) >= min_n])

        users_in_circle = filt_df.shape[0]
        number_of_circles = filt_df['Cluster'].value_counts().shape[0]

        min_list.append(min_n)
        max_list.append(max_n)
        n_users_list.append(users_in_circle)
        n_circles_list.append(number_of_circles)

    print(min_list)
    print(max_list)
    print(n_users_list)
    print(n_circles_list)

    list_of_lists = [min_list, max_list, n_users_list, n_circles_list]
    var_names = ['Min_Count', 'Max_Count', 'Users_Count', 'Circles_Count']

    print(pd.DataFrame(list_of_lists, index=var_names)
          .transpose()
          .query('Circles_Count >= 1000')
          .sort_values(by='Circles_Count'))

    #     Min_Count  Max_Count  Users_Count  Circles_Count
    # 44         17        136        32564           1000
    # 90         16         56        26619           1026
    # 70         13         23        17980           1088
    # 24         15        117        34737           1234
    # 72         14         60        31696           1335
    # 0          11         73        43102           2171
    # 57          6        114        83456           7320
    # 84          5         37        88443          10710
    # 9           2         30       321185         111173

    # ========================================================================

    # Adding in min max parameters
    filt_df = (final_df
               .loc[final_df['Cluster'].map(final_df['Cluster']
                                            .value_counts()) <= 136])
    # Setting the minimum number of people allowed in a circle.
    filt_df = (filt_df
               .loc[final_df['Cluster'].map(final_df['Cluster']
                                            .value_counts()) >= 17])

    print(filt_df.shape)  # 32564 individuals
    print(filt_df['Cluster'].value_counts().shape)  # In 1000 circles
    print(filt_df['Cluster'].unique())

    # What do the people in each cluster look like?
    print(filt_df['Cluster'].unique())  # Getting the individual cluster ints
    print(filt_df['Cluster'].value_counts())  # Counting n user in each circle
    print(filt_df.loc[filt_df['Cluster'] == 8])  # Checking circle 8

    #         Cluster                    Source           Target
    # 19            8    David_Ramsay_Clendenin  David_J._Eicher
    # 20            8           David_J._Eicher        Brian_May
    # 31            8              Arthur_Ducat  David_J._Eicher
    # 173           8          Adam_Gale_Malloy  David_J._Eicher
    # 348           8        William_Sooy_Smith  David_J._Eicher
    #          ...                       ...              ...
    # 12862         8  Maxwell_Van_Zandt_Woo...  David_J._Eicher
    # 12863         8        John_C._C._Sanders  David_J._Eicher
    # 14001         8            Thomas_W._Hyde  David_J._Eicher
    # 14403         8      William_Paul_Roberts  David_J._Eicher
    # 516121        8            John_H._Eicher  David_J._Eicher

    # American men involved in the civil war.
    # The key node in this directed sub-graph/community is David John Eicher
    # who is a historian who has researched and written extensively on the
    # American Civil War. Therefore, he edge is connected to the other nodes
    # but not vice-versa. He is also connected to Brian May as he is an
    # astrpohicist (and member of Queen). David ZEicher is also the editor of
    # Astronomy magazine and a popularizer of astronomy.

    # https://en.wikipedia.org/wiki/David_J._Eicher

    # What about a largest circle?
    print(filt_df.loc[filt_df['Cluster'] == 52])

    #          Cluster                Source                 Target
    # 270           52       John_B._Manning       Grover_Cleveland
    # 273           52      Grover_Cleveland      Adlai_Stevenson_I
    # 944           52      Robert_S._Kelley      Benjamin_Harrison
    # 1053          52       Boyd_Winchester      Elisha_Standiford
    # 1508          52  James_Barker_Edmonds       Grover_Cleveland
    #           ...                   ...                    ...
    # 201420        52         Pauline_Sabin       Grover_Cleveland
    # 310343        52        Albert_Sadacca       Grover_Cleveland
    # 372819        52     Arthur_A._Kimball   Dwight_D._Eisenhower
    # 1201438       52            Ivan_Eland         George_W._Bush
    # 1216555       52      Dorothy_Straight  Dorothy_Payne_Whitney

    # American presidents, politicians, military personal, defence analysts











    # SANITY CHECK DIRECTED AND UNDIRECTED GRAPHS WITH A MATRIXPLOT

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


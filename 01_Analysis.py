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
import matplotlib.pyplot as plt
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
    # Taking the pandas dataframe and parsing it into a undirected networkx
    # graph
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
    # pandas dataframe. I am doing this so once I setup a directed graph in the
    # next section I can go through and find those that are included in the
    # directed graph but not in the undirected graph here. This will then give
    # me a dataframe of names for just those with directed edges.
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
    # Sanity check on this stratifying strategy in networkx
    # =========================================================================
    # Take the adjacency matrix in a scipy sparse/numpy format, subtract its
    # transpose, and mask the resultant matrix to get the absolute values.
    # This will return the directed edges which can be used to sanity check
    # the stratification that has been done in networkx.

    # Now setting up a networkx graph object only using the final directed
    # edge subset and exporting as a graphml file for analysis in
    # scikit-network.
    directed_wiki_graph = nx.from_pandas_edgelist(df=final_dir_edge_df,
                                                  source='Source',
                                                  target='Target',
                                                  create_using=nx.DiGraph())
    print(type(directed_wiki_graph))  # DiGraph
    print(nx.info(directed_wiki_graph))
    nx.write_graphml(directed_wiki_graph, path='directed_wiki_graph.graphml')

    # Type: DiGraph
    # Number of nodes: 451386
    # Number of edges: 501784
    # Average in degree:   1.1117
    # Average out degree:   1.1117

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

    # Getting the cluster labels
    unique_labels, counts = np.unique(labels, return_counts=True)
    print(unique_labels)
    print(len(unique_labels))  # 20151 unique clusters (circles in this case)
    print(counts)  # Counts of users in each circle

    # Assessing the overall modularity of all the clusters
    modularity(undir_adjacency, labels)  # 0.041381350642334946

    # Overall this value is very low, however, modularity suffers
    # from a resolution limit and is unable to detect small communities as
    # are present in our network.

    # Now I am getting the individual nodes from the undirected networkx
    # graph and the clusters from the propagation model and creating a
    # dataframe that stores both together.
    node_series = pd.Series(undir_wiki_graph.nodes(), name='Nodes')
    clust_labels = pd.Series(labels, name='Cluster')
    nodes_labels = pd.concat([node_series, clust_labels], axis=1)
    print(node_series)
    print(clust_labels)
    print(nodes_labels)
    print(nodes_labels.shape)
    print(nodes_labels['Nodes'].value_counts())

    # Now I am creating a final df that contains the original edge list and now
    # has the the cluster labels merged in
    final_df = (pd.merge(left=nodes_labels,
                         right=parsed_df,
                         left_on='Nodes',
                         right_on='Source')
                .drop('Nodes', axis=1))
    print(final_df)
    print(final_df.shape)  # (4353922, 3)

    #          Cluster                   Source                    Target
    # 0              0      Andreas_Leigh_Aabel            Anders_Krogvig
    # 9              0           Anders_Krogvig              Gerhard_Gran
    # 18            12  Benjamin_Vaughan_Abbott             Austin_Abbott
    # 21            12            Austin_Abbott   Benjamin_Vaughan_Abbott
    # 24            12         Burroughs_Abbott             James_H._Kyle
    #           ...                      ...                       ...
    # 4353899       12          Lara_Wollington            Chloe_Hawthorn
    # 4353900       12              Leanne_Wong            Aleah_Finnegan
    # 4353918       12              Hudson_Yang               Eddie_Huang
    # 4353920       12              Carissa_Yip  Alexander_Ivanov_(che...
    # 4353921       12                YungManny                 Q_Da_Fool

    # Now filtering down to smaller circles. This is because we want to
    # introduce as many people to circles as possible without creating circles
    # that feel wierd.

    # Filtering out duplicates in the long dataframe to avoid people ending up
    # being counted multiple times in a cluster

    full_long_df = final_df  # First keeping full long df
    print(final_df.shape)  # (4353922, 3)
    print(final_df         # (731070, 3)  # With only individual node names
          .loc[final_df['Source']
               .duplicated(keep='first') == False].shape)

    final_df = (final_df
                .loc[final_df['Source']
                     .duplicated(keep='first') == False])

    # ========================================================================
    # Selecting the 10,000 circles
    # ========================================================================
    # The first approach is to randomly sample different combinations of 10,000
    # circles and look at the number of users in them.

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
    print(dfs_list[0].shape)  # (705868, 3)

    # Now let's create a dataframe containing the pseudo-random number used for
    # sampling, the number of users in each cluster, and a column to confirm
    # that it is 10,000 clusters that have been sampled.
    list_of_lists = [seed_n_list, n_users_list, n_circles_list]
    col_names = ['Seeds', 'N_Users', 'N_Circles']
    print(pd.DataFrame(list_of_lists, index=col_names)
          .transpose()
          .sort_values(by='N_Users', ascending=False))

    # As we can see the random sample with seed 2 returns the most number of
    # users.
    #    Seeds  N_Users  N_Circles
    # 1      2   709695      10000
    # 2      3   707878      10000
    # 4      5   707848      10000
    # 3      4   707728      10000
    # 7      8   707276      10000
    # 0      1   705868      10000
    # 5      6    25810      10000
    # 6      7    23782      10000
    # 8      9    20106      10000

    # For now, let's just take the 10,000 circles with the max number of users,
    # This corresponds to a seed = 2.
    clusters = (pd.Series(final_df['Cluster']
                          .value_counts()
                          .index)
                .sample(10000, random_state=2)  # Sampling the df
                .tolist())  # Getting the cluster names in a list

    # Extracting the 10,000 clusters from the full long edge list dataframe
    filt_df = full_long_df.loc[full_long_df['Cluster'].isin(clusters)]
    print(filt_df.shape)  # (4315553, 3)
    print(filt_df['Cluster'].value_counts().shape)  # (10000,)
    print(filt_df
          .groupby('Cluster')
          .count()
          .drop('Target', axis=1)
          .sort_values(by='Source', ascending=False)[0:100])  # (10000, 2)

    # Waaaaaaayy too many people in some of the circles
    #           Source
    # Cluster
    # 12       4239667
    # 11399      32845
    # 0           4953
    # 6806        3028
    # 4387        2750
    #           ...
    # 9759          28
    # 10808         27
    # 6838          27
    # 10811         27
    # 8297          27
    print(filt_df)  # Inspecting the df with the 10,000 circles

    #          Cluster               Source                    Target
    # 0              0  Andreas_Leigh_Aabel            Anders_Krogvig
    # 1              0  Andreas_Leigh_Aabel         Carl_Oscar_Munthe
    # 2              0  Andreas_Leigh_Aabel              Gerhard_Gran
    # 3              0  Andreas_Leigh_Aabel            Gerhard_Munthe
    # 4              0  Andreas_Leigh_Aabel    Hartvig_Andreas_Munthe
    #           ...                  ...                       ...
    # 4353917       12          Leanne_Wong           Viktoria_Komova
    # 4353918       12          Hudson_Yang               Eddie_Huang
    # 4353919       12          Hudson_Yang                 Jeff_Yang
    # 4353920       12          Carissa_Yip  Alexander_Ivanov_(che...
    # 4353921       12            YungManny                 Q_Da_Fool

    print(filt_df.loc[filt_df['Cluster'] == 10808])
    #          Cluster                Source            Target
    # 2861080    10808         Annelies_Maas  Conny_van_Bentum
    # 2861081    10808         Annelies_Maas    Reggie_de_Jong
    # 2861082    10808         Annelies_Maas  Wilma_van_Velsen
    # 3012189    10808      Jolanda_de_Rover    Kira_Toussaint
    # 3012190    10808        Kira_Toussaint  Jolanda_de_Rover
    #           ...                   ...               ...
    # 3345719    10808            Linda_Moes  Conny_van_Bentum
    # 3345720    10808            Linda_Moes  Jolanda_de_Rover
    # 3345721    10808            Linda_Moes   Karin_Brienesse
    # 3345722    10808            Linda_Moes       Kira_Bulten
    # 3349567    10808  Diana_van_der_Plaats  Conny_van_Bentum

    # Female European swimmers

    # ========================================================================
    # Alternative solutions
    # ========================================================================
    # The next solution would be to take a much larger sample of each 10k
    # clusters and work in something like a modularity metric. From here we
    # could calculate the modularity for each subset of the 10k clusters.
    # Following, we could standardise the n users and modularity scores
    # across the sub samples, sum them together, and then re-standardise them
    # into a scale free composite score. We could then find the sub-samples
    # that have the highest standard deviations from the mean. Alternatively,
    # we could first subset those who meet a modularity criterion across the
    # 10,000 clusters and go from there.

    # Another approach is to a priori set the maximum and minimum
    # number of people allowed in a circle. For example, we could set the
    # minimum number of people to 2, as a social circle with only 1 person in
    # it is not a useful circle. In addition, we could set the max value to say
    # 150. This could represent a large friends network, for example, a group
    # of people that all attended a yoga teacher training event or a community
    # of people all involved in a certain type of meditation practice.

    # By a-priori setting the lower and upper bounds of circle n membership,
    # we can now conduct a random search on different lower and upper bound
    # combinations and use these values to stratify the network. Following,
    # we can subset all of these networks and inspect the number of circles in
    # each. Following, we can take all of those with around 10,000 circles and
    # then find those with the most amount of users and the highest modularity.
    # ========================================================================

    # Random search approach
    seed = np.random.seed(10)
    min_dist = np.random.randint(low=2, high=100, size=200)
    max_dist = np.random.randint(low=2, high=100, size=200)
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
          .query('Circles_Count <= 10000 & Circles_Count >= 100')
          .sort_values(by='Circles_Count', ascending=False))

    #      Min_Count  Max_Count  Users_Count  Circles_Count
    # 9            2         47        27676          10000
    # 124          2         21        26947           9976
    # 97           2         15        26416           9946
    # 190          4         14         7385           1334
    # 180          5         48         6340            776
    # ..         ...        ...          ...            ...
    # 176         11         86         3116            153
    # 181         11         47         2376            142
    # 0           11         44         2284            140
    # 144         13         85         2601            108
    # 45          13         60         2016            100

    # Filtering based off of the searched max and min values
    filt_df = (
        final_df
        .loc[final_df['Cluster'].map(final_df['Cluster']
                                     .value_counts()) <= 47]
    )
    # Setting the minimum number of people allowed in a circle.
    filt_df = (
        filt_df
        .loc[final_df['Cluster'].map(final_df['Cluster']
                                     .value_counts()) >= 2]
    )

    # What do the people in each cluster look like?
    print(filt_df['Cluster'].unique())  # Getting the individual cluster ints
    print(filt_df['Cluster'].value_counts())  # Counting n user in each circle
    print(filt_df.loc[filt_df['Cluster'] == 9645])  # Checking circle 5
    print(filt_df.shape)

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

    # We can see that they are all European fencers

    # Now getting the long form df with these 10,000 circles and all of the
    # edges for each node.
    circle_list = filt_df['Cluster'].unique().tolist()
    print(circle_list)

    # Sub-setting
    ten_k_circles_long_df = (
        full_long_df
        .loc[full_long_df['Cluster']
                          .isin(filt_df['Cluster']) == True]
    )

    print(ten_k_circles_long_df['Source'].value_counts())  # Checking long form
    # Joe_Schofield          51
    # Ana_Maria_Popescu      42
    # Hamid_Sourian          28
    # Viktor_Chegin          27
    # Kevin_Borl%C3%A9e      20
    #                        ..
    # Annie_Vitelli           1
    # Stein_Lier-Hansen       1
    # H%C3%A9ctor_del_Mar     1
    # John_Ronald_Lidster     1
    # Salvatore_Giardina      1

    print(ten_k_circles_long_df.shape)  # (45955, 3)
    ten_k_circles_long_df.to_csv('ten_k_circles_long_df.csv')
    (ten_k_circles_long_df.drop('Cluster', axis=1)
     .to_csv('ten_k_circles_long_df.csv'))

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

    modularity(dir_adjacency, labels)  # 0.4943768068371678
    # Much better modularity

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
    print(final_df.shape)
    print(final_df['Source'].value_counts())

    # Ned_Chaillet         416
    # Harold_Hecht         371
    # Leander_Paes         341
    # Bernard_Alane        314
    # Annie_Cordy          272
    #                     ...
    # Donald_Pratt           1
    # Peter_Linz             1
    # William_B._Ellern      1
    # Hsi_Tseng_Tsiang       1
    # Hans_Lampe             1

    final_df.to_csv('undirected_clust_df_test.csv')

    #          Cluster                    Source                    Target
    # 0              0      Joaqu%C3%ADn_Riascos             Santos_Acosta
    # 1              0             Santos_Acosta      Joaqu%C3%ADn_Riascos
    # 2              1              James_Masson  Benjamin_Allen_(Canad...
    # 3              1  Benjamin_Allen_(Canad...              James_Masson
    # 4              1  Benjamin_Allen_(Canad...     Samuel_Johnathan_Lane
    #           ...                       ...                       ...
    # 2618517   176056               Emma_Spence          No%C3%A9mi_Makra
    # 2618518   176056               Emma_Spence           Rose-Kaying_Woo
    # 2618519   176056               Emma_Spence                Teja_Belak
    # 2618520   207812          No%C3%A9mi_Makra          Andreea_Munteanu
    # 2618521   207812          No%C3%A9mi_Makra          Maria_Kharenkova

    # Filtering out duplicates in the long dataframe to avoid people ending up
    # being counted multiple times in a cluster
    full_long_df = final_df  # Keeping long form to subset later
    final_df = (final_df
                .loc[final_df['Source']
                     .duplicated(keep='first') == False])
    print(final_df.shape)

    # =========================================================================

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
                          .sample(1000, random_state=n)  # Iterate over this
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
    print(dfs_list[0].shape)  # (21204, 3)

    # Now let's create a dataframe containing the pseudo-random number used for
    # sampling, the number of users in each cluster, and a column to confirm
    # that it is 1000 circles that have been sampled.
    list_of_lists = [seed_n_list, n_users_list, n_circles_list]
    col_names = ['Seeds', 'N_Users', 'N_Circles']
    print(pd.DataFrame(list_of_lists, index=col_names)
          .transpose()
          .sort_values(by='N_Users', ascending=False))

    #    Seeds  N_Users  N_Circles
    # 2      3     2413       1000
    # 7      8     2212       1000
    # 6      7     2197       1000
    # 4      5     2117       1000
    # 5      6     2111       1000
    # 1      2     2109       1000
    # 8      9     2106       1000
    # 3      4     2057       1000
    # 0      1     2014       1000

    # For now, let's just take the 10,000 circles with the max number of users,
    # This corresponds to a seed = 3.
    dir_clusters = (pd.Series(final_df['Cluster']
                              .value_counts()
                              .index)
                    .sample(1000, random_state=3)  # Sampling the df
                    .tolist())  # Getting the cluster names in a list

    # Extracting the 10,000 clusters from the full long edge list dataframe
    filt_df = full_long_df.loc[full_long_df['Cluster'].isin(dir_clusters)]
    print(filt_df.shape)  # (12987, 3)
    print(filt_df['Cluster'].value_counts().shape)  # (1000,)
    print(filt_df
          .groupby('Cluster')
          .count()
          .drop('Target', axis=1)
          .sort_values(by='Source', ascending=False)[0:100])  # (10000, 2)

    #          Source
    # Cluster
    # 1981        985
    # 1337        263
    # 8854        152
    # 20338       147
    # 179977      138
    #          ...
    # 177868       28
    # 176750       28
    # 152698       28
    # 89720        28
    # 161072       28

    # Let's inspect some of these circles and see if they make sense
    print(filt_df.loc[filt_df['Cluster'] == 1981])

    #          Cluster                 Source                 Target
    # 21306       1981  Franklin_D._Roosevelt           Adolf_Hitler
    # 21307       1981  Franklin_D._Roosevelt               Al_Smith
    # 21308       1981  Franklin_D._Roosevelt        Albert_Einstein
    # 21309       1981  Franklin_D._Roosevelt        Albert_Ottinger
    # 21310       1981  Franklin_D._Roosevelt             Alf_Landon
    #           ...                    ...                    ...
    # 1700347     1981           Kesha_Rogers           Alvin_Greene
    # 1700348     1981           Kesha_Rogers           Barack_Obama
    # 1700349     1981           Kesha_Rogers  Franklin_D._Roosevelt
    # 1700350     1981           Kesha_Rogers        Lyndon_LaRouche
    # 1700351     1981           Kesha_Rogers             Pete_Olson

    # What about a smaller circle?
    print(filt_df['Cluster'].value_counts()[0:50])
    # 1981      985
    # 1337      263
    # 8854      152
    # 20338     147
    # 179977    138
    #          ...
    # 149924     44
    # 141558     40
    # 60827      40
    # 53808      39
    # 140997     39

    print(filt_df.loc[filt_df['Cluster'] == 179977])
    #          Cluster         Source                    Target
    # 2297171   179977  Stan_Wawrinka  %C3%89douard_Roger-Va...
    # 2297172   179977  Stan_Wawrinka          Adrian_Mannarino
    # 2297173   179977  Stan_Wawrinka              Albert_Costa
    # 2297174   179977  Stan_Wawrinka           Alejandro_Falla
    # 2297175   179977  Stan_Wawrinka      Alessandro_Giannessi
    #           ...            ...                       ...
    # 2297304   179977  Stan_Wawrinka            Vasek_Pospisil
    # 2297305   179977  Stan_Wawrinka       Victor_H%C4%83nescu
    # 2297306   179977  Stan_Wawrinka            Viktor_Troicki
    # 2297307   179977  Stan_Wawrinka            Xavier_Malisse
    # 2297308   179977  Stan_Wawrinka        Yoshihito_Nishioka

    # Successful male tennis players
    filt_df.loc[filt_df['Cluster'] == 179977][['Source', 'Target']].to_csv(
        'directed_graph_tennis_play_subset.csv'
    )


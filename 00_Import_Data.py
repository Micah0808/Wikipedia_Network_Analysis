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

    # Now cleaning up this date dataframe into a usable format.
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


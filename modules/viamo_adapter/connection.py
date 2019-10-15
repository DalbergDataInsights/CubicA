import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pymysql
from sqlalchemy import create_engine
from tqdm import tqdm


class Connection:
    """
    Singleton connection object that will talk to the db
    The class is singleton for the purposes of not openning
    connection every userobject
    """
    
    __instance = None
    data_path = os.path.join(os.getcwd(), 'data/extracts/')

    # Making the class singleton
    def __new__(cls,
                username: str = None, password: str = None,
                host: str = '10.2.0.3',
                port: int = 3306,
                dialect: str = 'mysql',
                driver: str = 'pymysql',
                database: str = None,
                chunksize=50000, new=False):
        if Connection.__instance is None or new:
            assert(username is not None), \
                    'Please specify username when initializing the connection for the first time'
            assert(password is not None), \
                    'Please specify the password when initializaing the connection for the first time'
            Connection.__instance = object.__new__(cls)
            Connection.__instance.chunksize = chunksize
            # Connection.__instance.engine = pymysql.connect(
            #                         host=host,
            #                         user=username,
            #                         passwd=password,
            #                         port=3306)
            # Connection.__instance.cursor = Connection.__instance.engine.cursor()

            # Switching to sqlalchemy db connector
            if driver is not None:
                driver = '+' + driver
            else:
                driver = ''
            if database is not None:
                database = '/' + database
            else:
                database = ''

            initiation = f'{dialect}{driver}://{username}:{password}@{host}:{port}{database}'
            Connection.__instance.engine = create_engine(initiation)
            Connection.__instance.cursor = Connection.__instance.engine.connect()
        return Connection.__instance
        
    def execute_query(self, query, one=True):
        """Returns first result of a query"""
        if one:
            return self.cursor.execute(query).fetchone()
        else:
            result = [x[0] for x in list(self.cursor.execute(query).fetchall())]
            return result

            
class Adaptor:

    def __init__(self, *args, **kwargs):
        self.connection = Connection(**kwargs)
        self._content_blocks = None

    @property
    def content_blocks(self):
        if self._content_blocks is None:
            self._content_blocks = pd.read_sql("SELECT * FROM cubica.content_blocks",
                                             con=self.connection.cursor)
        else:
            return self._content_blocks
    
    @content_blocks.setter
    def content_blocks(self, value):
        self._content_blocks = value

    def refresh_content_blocks(
                            self,
                            drop_titles=None,
                            to_file=False,
                            no_start_blocks=False,
                            regex=False,
                            to_db=False) -> (pd.DataFrame, dict, dict):
        """
        Refreshing content_blocks from the database
        Arguments:
            drop_titles {list} -- List of str to pop from content_blocks dict
            to_file {bool}
            no_start_blocks {bool}
            regex {bool}
            to_db {bool}

        Returns:
            DataFrame with content_blocks_ids and DDI_ids
            Poped Keys Dicst -- Useful for debugging and sanity checks
            content_blocks ID Dict -- Dictionary with title
        """
        # Aproximate non-content messages
        if drop_titles is None:
            drop_titles = [
                        "thank you",
                        "goodbye",
                        "welcome",
                        "sorry",
                        "call",
                        "no credit",
                        "registration complete",
                        "registered",
                        "intro"]

        def find_title(long_title: str) -> str or None:
            """
            Function to help with messy titles

            Arguments:
                long_title {str} -- Long title to clean

            Returns:
                str or None -- Cleaned title if possible
            """
            import re
            title = re.match(r'{"title"\:"[\d.,]*(.*)","allow.*', long_title)
            if title:
                return title.group(1).strip().lower()
            else:
                return None

        def drop_numbers(title: str) -> str:
            """
            Function to help with title cleaning

            Arguments:
                title {str} -- Messy title to clean

            Returns:
                str -- Cleaned title
            """

            if title:
                if title[0] in '0123456789' and ' ' in title:
                    title = title[title.index(' '):].lstrip()
            return strip_non_ascii(title).lower()

        def strip_non_ascii(s):
            return "".join(i for i in s if ord(i) < 126 and ord(i) > 31)
    
        # 1. Get all non-starting blocks

        # choose all blocks from the db
        blocks = pd.read_sql('select * from voto_app.blocks;', con=self.connection.cursor)
        # get only message blocks
        mblocks = blocks[blocks.class_name == 'MessageBlock']
        del blocks
    
        if no_start_blocks:
            # handle start blocks
            start_blocks = pd.read_sql('select start_block_id from voto_app.block_connections;',
                                        con=self.connection.cursor)
            start_blocks = start_blocks.start_block_id
            
        else:
            start_blocks = pd.read_sql('select distinct(starting_block_id) from voto_app.trees;',
                                        con=self.connection.cursor)
            start_blocks = start_blocks.starting_block_id

        start_blocks = start_blocks.drop_duplicates()
        start_blocks = start_blocks.dropna()
        content_blocks = mblocks[~mblocks.id.isin(start_blocks)]
        del mblocks
        
        # filter titles
        if regex:
            titles = content_blocks['details'].apply(find_title)
        else:
            titles = content_blocks['details'].apply(json.loads)
            titles = [x.get('title') for x in list(titles)]
            new_titles = []
            for title in titles:
                title = drop_numbers(title)
                new_titles.append(title)
            titles = new_titles.copy()
            del new_titles
        
        content_blocks['title'] = titles
        content_blocks_ids = content_blocks.groupby("title")['id'].apply(list)
        
        del content_blocks

        # 2. Pop non-content blocks
        to_pop = []
        # form to_pop list to return it to user (sanity check)
        for key in content_blocks_ids.keys():
            for pop in drop_titles:
                if pop.lower() in key.lower() and key not in to_pop:
                    to_pop.append(key)
                if key.lower().startswith('end') and key not in to_pop:
                    to_pop.append(key)

        # popping not content_blocks
        for tp in to_pop:
            content_blocks_ids.pop(tp)

        content_blocks_arr = []
        for key, value in content_blocks_ids.items():
            oldest = min(value)
            if not key == '':
                for index in value:
                    content_blocks_table = {}
                    # index is a block id
                    content_blocks_table['block_id'] = index
                    # key is a title
                    content_blocks_table['title'] = key
                    # new key is the oldest id found (or lowest)
                    content_blocks_table['ddi_id'] = "ddi_content_" + str(oldest)
                    content_blocks_arr.append(content_blocks_table)
                
        output = pd.DataFrame(content_blocks_arr)
        output = output.dropna()

        # writing to files

        if to_file:
            with open(os.path.join(os.getcwd(), 
                    'data\\extracts\\content_blocks_poped.json'), 'w') as pop:
                json.dump(to_pop, pop)
                pop.close()
            with open(os.path.join(os.getcwd(), 
                    'data\\extracts\\content_blocks_ids.json'), 'w') as f:
                json.dump(content_blocks_ids.to_dict(), f)
                f.close()
            output.to_csv(os.path.join(os.getcwd(), 
                    'data\\extracts\\content_blocks.csv'), index=False)
        
        # write to db
        if to_db:
            output.to_sql('content_blocks', con=self.connection.cursor,
                        schema='cubica', index=False, if_exists='replace')

        self.content_blocks = output

        return content_blocks_ids, to_pop

    # Untested
    # def get_mb_interactions(self,
    #                         fields_to_query: list = [
    #                                                 "block_id",
    #                                                 "subscriber_id",
    #                                                 "audio_percent_listened"],
    #                         overwrite=False):
    #     """
    #     Refresh MessageBlocks interactions file
        
    #     Returns:
    #         Filename with block_interactions extract
    #     """
    #     # query db only if today's file doesnt exist
    #     filename = os.path.join(os.getcwd(),
    #                             f'data\\extracts\\interactions_{datetime.now().strftime("%Y%d%m")}.csv')
    #     if not os.path.exists(filename) or overwrite:
    #         # We are going to iterate tree by tree to make it more efficient
    #         resulst = self.connection.execute.execute('select id from voto_app.trees').fetchall()
    #         trees = [row[0] for row in resulst]

    #         fields = ','.join(fields_to_query)
        
    #         for tree in tqdm(trees):
    #             self.cursor.execute(f"""SELECT {fields} 
    #                                 FROM voto_app.block_interactions
    #                                 WHERE tree_id = {tree} 
    #                                 AND class_name = 'MessageBlock';""")
    #             if self.cursor.rowcount > 0:
    #                 st = pd.DataFrame(list(self.cursor.fetchall()), columns=fields_to_query)
    #                 if not st.empty:
    #                     # get message block
    #                     if not os.path.exists(filename):
    #                         st.to_csv(filename, index=False)
    #                     else:
    #                         # append to a table of results
    #                         st.to_csv(filename, mode='a', index=False, header=False)
    #                 # to free memory for next big dataframe
    #                 del st
    #     return filename

    # def filter_mb_interactions(self, filename, content_blocks=None):

    #     if content_blocks is None:
    #         content_blocks = pd.read_csv(self.data_path+'content_blocks.csv')
    #     header = True
    #     for chunk in pd.read_csv(filename, chunksize=self.chunksize):
    #         # join on block_id
    #         block_interactions = chunk.set_index('block_id') \
    #                                 .join(content_blocks.set_index('block_id'),
    #                                 on='block_id', how='inner')

    #         block_interactions.to_csv(self.data_path
    #                                 + f'content_block_interactions_{datetime.now().strftime("%Y%d%m")}.csv',
    #                                 mode='a+', header=header)
    #         header = False

import pandas as pd
import os
import json

from tqdm import tqdm

from modules.viamo import connection

class RatingMatrix:


    data_path = os.path.join(os.getcwd(), 'data/extracts/')

    def __init__(self):
        self.matrix = None
        self.users = {}


    @staticmethod
    def get_subscriber_interactions_vector(sub_id,
                                            content_blocks=None,
                                            mtype='counts',
                                            full=False):
        """
        Gets you the vector in subscriber X ddi_block_id shape
        with selected type of aggregation
        
        Arguments:
            sub_id {str} -- Id of requested subscriber
        
        Keyword Arguments:
            content_blocks {DataFrame} -- Content blocks dataframe (default: {None})
            mtype {strin} -- Type of aggregation ('counts', '1/0', 'listened')
                                                    (default: {'counts'})
            full {bool} -- Whther the vector should contain all headers
                        or just the blocks user interacted with (default: {False})
        
        Returns:
            DataFrame -- Vector of user interactions
        """

        if content_blocks is None:
            content_blocks = pd.read_csv(RatingMatrix.data_path+'content_blocks.csv')
            content_blocks = content_blocks.dropna()
        # query the block_interactions
        columns = ['block_id', 'audio_percent_listened']

        col_query = ','.join(columns)

        con = connection.Connection()

        sub_interactions = pd.read_sql(f"""SELECT {col_query} 
                                        FROM voto_app.block_interactions 
                                        WHERE subscriber_id = {sub_id} 
                                        AND class_name = 'MessageBlock';""",
                                        con = con.cursor)

        sub_content = sub_interactions.set_index('block_id') \
                                    .join(content_blocks.set_index('block_id'),
                                    on='block_id', how='inner',
                                    lsuffix='_sub', rsuffix='_leaf')

        if not sub_content.empty: 
            
            if mtype == 'counts':
                count = sub_content.groupby('ddi_id').agg({'title': 'count'})
                if full:
                    gcontent_blocks = content_blocks[['ddi_id', 'block_id']] \
                                            .groupby(by='ddi_id').agg('count')
                    count = gcontent_blocks.join(count, on='ddi_id', how='left').fillna(0)
                count.title = pd.to_numeric(count.title, downcast='integer')
                count = count.reset_index()
                count['sub'] = sub_id
                matrix = count.pivot(columns='ddi_id', values='title', index='sub')
                del count
            del sub_interactions
            return matrix
        
        else:
            del sub_interactions
            return None

    @staticmethod
    def get_matrix_headers(content_blocks=None):
        """Simple function that will return you an empty dataframe
            with headers of current content blocks (or content blocks that are passes)"""

        con = connection.Connection()
        if content_blocks is None:
            content_blocks = pd.read_sql("""SELECT * FROM cubica.content_blocks""",
                                        con = con.engine)
            content_blocks = content_blocks.dropna()

        gcontent_blocks = content_blocks[['ddi_id', 'block_id']].groupby(by='ddi_id').agg('count')
        
        gcontent_blocks['sub'] = 0
        gcontent_blocks['val'] = 0
        gcontent_blocks = gcontent_blocks.reset_index()
        pcontent_blocks = gcontent_blocks.pivot(columns = 'ddi_id', index='sub', values = 'val')

        return pcontent_blocks

    def add_user(self, user):
        
        self.users[user._id] = user

    def get_user(self,sub_id):
        return self.users.get(sub_id, None)

    def generate_matrix(self):
        pass
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
            Connection.__instance.engine = \
            create_engine(initiation)
            Connection.__instance.cursor = Connection.__instance.engine.connect()
        return Connection.__instance
        
    def execute_query(self, query, one=True):
        """Returns first result of a query"""
        if one:
            return self.cursor.execute(query).fetchone()
        else:
            return self.cursor.execute(query).fetchall()

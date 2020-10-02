
import pandas as pd
from datetime import datetime

import os

cf = os.getcwd()


from modules.viamo import connection


con = connection.Connection(username='', password='')

con.refresh_leafs()

filename = con.get_mb_interactions(overwrite=True)

con.get_matrix(filename)

print("")

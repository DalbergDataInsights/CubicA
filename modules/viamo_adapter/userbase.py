from modules.viamo import rmatrix
from modules.viamo import connection

class User():

    @property
    def rating_vector(self):
        if self._rating_vector is None:
            self.__get_interactions_vector()
        return self._rating_vector
    
    @property
    def is_empty(self):
        if self.rating_vector is None:
            return True
        return False


    def __init__(self, sub_id):
        self._id = sub_id
        self._connection = connection.Connection()
        self._rating_vector = None
        # Check if user is flagged as active
        active, self.preferred_content = self._connection.execute_query(
            f"""SELECT active, preferred_content_type 
            FROM voto_app.subscribers
            WHERE id = {self._id};"""
        )
        self.is_active = True if active else False
        
    def __get_interactions_vector(self, content_blocks=None, mtype='counts', full=False):
        self._rating_vector = rmatrix.RatingMatrix.get_subscriber_interactions_vector(self._id, 
                                                                            content_blocks)
    
    # !TODO 

    # Getter of user stats
    # getter of user profile type
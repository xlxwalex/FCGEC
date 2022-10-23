class Point(object):
    """Point Class"""

    def __init__(self, index : int, token : str, offset : int = 0):
        self.point_index = index
        self.token = token
        self.offset = offset

    def __str__(self):
        return "(%d:%s[%d])" % (self.point_index, self.token, self.offset)

    def __repr__(self):
        return str(self)
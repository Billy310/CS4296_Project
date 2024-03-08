class User:
    def __init__(self, name, password,socketid):
        self.name = name
        self.password = password
        self.socketid = socketid

    def get_name(self):
        return self.name

    def get_password(self):
        return self.password
    
    def get_socketid(self):
        return self.socketid

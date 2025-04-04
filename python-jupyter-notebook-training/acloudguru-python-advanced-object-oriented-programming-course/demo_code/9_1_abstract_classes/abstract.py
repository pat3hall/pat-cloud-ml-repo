from abc import ABC, abstractmethod, abstractclassmethod, abstractstaticmethod

class APIClient(ABC):
    def __init__(self, domain):
        self.domain = domain

    @abstractmethod
    def get(self, url):
        pass

    @abstractclassmethod
    def default_headers(cls):
        pass

    @abstractstaticmethod
    def static_option():
        pass

class MyClient(APIClient):
    pass


        
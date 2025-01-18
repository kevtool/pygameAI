from pygameai import pygameAI

class Algorithm:
    def __init__(self, env):
        assert issubclass(type(env), pygameAI)
        self.env = env
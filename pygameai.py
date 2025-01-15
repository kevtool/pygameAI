class pygameAI():
    def __init__(self):
        pass

    def set_action_space(self, type, num):
        assert type in ['discrete', 'continuous']
        self.action_type = type
        if type == 'discrete':
            self.action_space = [i for i in range(num)]
        elif type == 'continuous':
            pass

    def set_obs_space(self, type, num):
        assert type in ['discrete', 'continuous']
        self.obs_type = type
        if type == 'discrete':
            self.obs_space = [i for i in range(num)]
        elif type == 'continuous':
            pass

    def reset(self):
        pass

    def step(self):
        pass

    def render(self):
        pass
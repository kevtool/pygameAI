class pygameAI():
    def __init__(self):
        pass

    def set_game_speed(self, game_speed):
        self.game_speed = game_speed

    def set_action_space(self, type, shape):
        assert type in ['discrete', 'continuous']
        self.action_type = type
        self.action_shape = shape

        if type == 'discrete':
            self.action_space = [i for i in range(shape)]
        elif type == 'continuous':
            pass

    def set_obs_space(self, type, shape):
        assert type in ['discrete', 'continuous']
        self.obs_type = type
        self.obs_shape = shape

    def reset(self):
        pass

    def step(self):
        pass

    def render(self):
        pass
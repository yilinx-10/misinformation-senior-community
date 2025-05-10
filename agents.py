from mesa import Agent

class NetworkAgent(Agent):
    #Define initiation of agents
    def __init__(
        self,
        model,
        cognitive_ability,
        digital_literacy,
        se_motivated,
        belief_scale
    ):
        super().__init__(model)
        self.cognitive_ability = cognitive_ability
        self.digital_literacy = digital_literacy
        self.se_motivated = se_motivated
        self.belief_scale = belief_scale
        self.received = [] # tuple of (from whom, Trust or Distrust)

    def check(self):
        # Check agent's belief_scale to determine
        # distrust and spread to all < distrust and spread to close < distrust <
        # ignore < trust < trust and spread to close < trust and spread to all 
        receivers_lst = []
        if self.belief_scale < - 0.66 or self.belief_scale > 0.66:
            receivers_lst = self.select_receiver('all')
        elif -0.66 <= self.belief_scale < -0.33 or  0.66 >= self.belief_scale > 0.33:
            receivers_lst = self.select_receiver('close')
        self.spread(receivers_lst)
        
    def select_receiver(self, mode):
        neighbors = self.model.grid.get_neighbors(self, include_center=False, radius = 1)
        if mode == "close":
            max_node = max(self.model.G[self], key=lambda node: self.model.G[1][node]['weight'])
            neighbors = list(max_node)
        return neighbors

    def spread(self, receivers_lst):
        for receiver in receivers_lst:
            receiver.received.append((self, self.belief_scale))
            if receiver.se_motivated:
                se_score = self.belief_scale * self.model.G[receiver][self]['weight']
            # Need to think more about how to construct the scale and scores
            receiver.belief_scale += receiver.digital_literacy*receiver.cognitive_ability*se_score

    def fact_checker(self):
        if self.random.random() > self.model.fact_checking_prob:
            if self.belief_scale < 0:
                self.belief_scale += 1 # Adjustments HERE PERHAPS?
                self.adjust_weight()

    def adjust_weight(self):
        for node, attitude in self.received:
            if attitude > 0:
                self.model.G[self][node]['weight'] *= self.model.confidence_deprecation_rate

    def step(self):
        self.check()
        self.spread()
        self.fact_checker()
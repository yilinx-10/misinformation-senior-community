from mesa import Agent

# To Do
# 1 Repetition's effect will make shifts in scale larger each round

class NetworkAgent(Agent):
    #Define initiation of agents
    def __init__(
        self,
        model,
        node,
        cognitive_ability,
        digital_literacy,
        se_motivated,
        belief_scale
    ):
        super().__init__(model)
        self.node = node
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
        neighbors = self.model.grid.get_neighbors(self.node, include_center = False, radius = 1)
        if mode == "close":
            max_weight = 0
            for neighbor in neighbors:
                if self.model.G.has_edge(neighbor.node, self.node):
                    weight = self.model.G[self.node][neighbor.node]['weight']
                    if weight > max_weight:
                        max_weight = weight
                        neighbors = [neighbor]
        return neighbors

    def spread(self, receivers_lst):
        for receiver in receivers_lst:
            if self.random.random() > 0.5:
                receiver.received.append((self.node, self.belief_scale))
                se_score = 1
                text_or_visual = 1
                authority = 1
                if receiver.se_motivated:
                    if self.model.G.has_edge(receiver.node, self.node):
                        se_score = 1.1 * self.model.G[receiver.node][self.node]['weight']
                # Need to think more about how to construct the scale and scores
                if self.model.info_format == "visual":
                    text_or_visual = 1.1
                if not self.se_motivated:
                    authority = 1.1
                receiver.belief_scale += receiver.digital_literacy*receiver.cognitive_ability*text_or_visual*se_score*authority
                receiver.belief_scale = min(receiver.belief_scale, 1)
                receiver.belief_scale = max(receiver.belief_scale, -1)

    def fact_checker(self):
        if self.random.random() > self.model.fact_checking_prob:
            if self.belief_scale > 0:
                self.belief_scale  *= 0.5# Adjustments HERE PERHAPS?
                self.adjust_weight()

    def adjust_weight(self):
        if self.received:
            for sender, attitude in self.received:
                if attitude > 0:
                    if self.model.G.has_edge(self.node, sender):
                        # ADD CONDITION TO ENABLE CHANGE ONLY IN WEIGHTED
                        self.model.G[self.node][sender]['weight'] *= self.model.confidence_deprecation_rate

    def step(self):
        self.check()
        self.fact_checker()
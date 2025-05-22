from mesa import Agent

# repetition increases general susceptibility  

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
                weight = self.model.G[self.node][neighbor.node]['weight']
                if weight > max_weight:
                    max_weight = weight
                    neighbors = [neighbor]
        return neighbors

    def spread(self, receivers_lst):
        for receiver in receivers_lst:
            if self.random.random() > 0.5: # by chance notice
                receiver.received.append((self.node, self.belief_scale))

                if self.belief_scale > 0 and self.random.random() < self.model.fact_checking_prob * receiver.digital_literacy:
                    receiver.belief_scale  = -1# Adjustments HERE PERHAPS?
                    receiver.adjust_weight()
                    break

                se_score = 0.5
                authority = 0.5

                if receiver.se_motivated: # only residents are

                    se_score += self.model.G[receiver.node][self.node]['weight']

                    if not self.se_motivated: # if sender is not resident
                        authority = 1

                # Independent thinking
                # If higher digital literacy higher cognitive ability, then 
                # 1) sender wrong belief: update own belief less
                # 2) sender correct belief: update own belief more
                if self.belief_scale > 0:
                    receiver.belief_scale += (1 - receiver.digital_literacy)*(1 - receiver.cognitive_ability)*se_score*authority
                elif self.belief_scale < 0:
                    receiver.belief_scale += receiver.digital_literacy*receiver.cognitive_ability*se_score*authority

                receiver.belief_scale = min(receiver.belief_scale, 1)
                receiver.belief_scale = max(receiver.belief_scale, -1)

    def adjust_weight(self):
        if self.received:
            for sender, attitude in self.received:
                if attitude > 0:
                    self.model.G[self.node][sender]['weight'] *= (1 - self.model.confidence_deprecation_rate)

    def step(self):
        self.check()
import copy
from hashlib import new
import random
import numpy as np
from player import Player


class Evolution:
    def __init__(self):
        self.game_mode = "Neuroevolution"

    def next_population_selection(self, players, num_players):
        """
        Gets list of previous and current players (μ + λ) and returns num_players number of players based on their
        fitness value.

        :param players: list of players in the previous generation
        :param num_players: number of players that we return
        """
       
        sorted_players=sorted(players,key=lambda x: x.fitness,reverse=True)
        f = open("stats.txt", "a")
        max_gen=sorted_players[0].fitness
        min_gen=sorted_players[len(sorted_players)-1].fitness
        avg_gen=0
        for i in sorted_players:
            avg_gen+=i.fitness
        avg_gen/=1.0*len(sorted_players)
        f.writelines([f"{max_gen}\n", f"{min_gen}\n",f"{avg_gen}\n"])
        f.close()


        method="top-k"
        if method=="top-k":
            return sorted_players[: num_players]
        elif method=="rw":
            probabilities=[]
            summ=0
            for x in sorted_players:
                summ+=x.fitness
            for j in sorted_players:
                probabilities.append(1.0*j.fitness/summ)
            result = np.random.choice(sorted_players,num_players,p=probabilities).tolist()
            return result
        elif method=="q-tournament":
            ans=[]
            Q=3
            for i in range(0,num_players):
                tmp=np.random.choice(sorted_players,Q).tolist()
                max_tmp=tmp[0]
                for k in tmp:
                    if k.fitness>max_tmp.fitness:
                        max_tmp=k
                ans.append(max_tmp)
            return ans
        

        

    def mutate(self,player):
        neural_network=player.nn
        weights=neural_network.weights
        biases=neural_network.biases
        for weight in weights:
            for i in range(0,weight.shape[0]):
                for j in range(0,weight.shape[1]):
                    if random.uniform(0,1)<0.1:
                        weight[i][j]+=np.random.normal(0,1)
        for bias in biases:
            for i in range(0,bias.shape[0]):
                if random.uniform(0,1)<0.1:
                    bias[i][0]+=np.random.normal(0,1)


    def recombination(self,player1,player2):
        neural_network1=player1.nn
        weights1=neural_network1.weights
        biases1=neural_network1.biases
        neural_network2=player2.nn
        weights2=neural_network2.weights
        biases2=neural_network2.biases

        if random.uniform(0,1)<0.4:
            alpha=random.uniform(0.25,0.75)
            for weight in range(0,len(weights1)):
                for i in range(0,weights1[weight].shape[0]):
                    for j in range(0,weights1[weight].shape[1]):
                        
                        weights1[weight][i][j]=(1-alpha)*weights1[weight][i][j] + alpha*weights2[weight][i][j]
                        weights2[weight][i][j]=alpha*weights1[weight][i][j] + (1-alpha)*weights2[weight][i][j]

            for bias in range(0,len(biases1)):
                for i in range(0,biases1[bias].shape[0]):
                    biases1[bias][i][0]=(1-alpha)*biases1[bias][i][0] + alpha * biases2[bias][i][0]
                    biases2[bias][i][0]=alpha*biases1[bias][i][0] + (1-alpha) * biases2[bias][i][0]
        else:
            return

    def generate_new_population(self, num_players, prev_players=None):
        """
        Gets survivors and returns a list containing num_players number of children.

        :param num_players: Length of returning list
        :param prev_players: List of survivors
        :return: A list of children
        """
        first_generation = prev_players is None
        if first_generation:
            return [Player(self.game_mode) for _ in range(num_players)]
        else:

            method="regular"
            if method=="regular":
                new_players=[]
                for i in range(0,num_players,2):
                    first_child=self.clone_player(prev_players[i])
                    second_child=self.clone_player(prev_players[i+1])
                    self.recombination(first_child,second_child)
                    self.mutate(first_child)
                    self.mutate(second_child)
                    new_players.append(first_child)
                    new_players.append(second_child)
                return new_players
            elif method=="rw":
                probabilities=[]
                summ=0
                for x in prev_players:
                    summ+=x.fitness
                for j in prev_players:
                    probabilities.append(1.0*j.fitness/summ)
                good_parents = np.random.choice(prev_players,num_players//2,p=probabilities).tolist()
                new_players=[]
                for i in range(0,num_players//2):
                    tmp=np.random.choice(good_parents,2).tolist()
                    first_child=self.clone_player(tmp[0])
                    second_child=self.clone_player(tmp[1])
                    self.recombination(first_child,second_child)
                    self.mutate(first_child)
                    self.mutate(second_child)
                    new_players.append(first_child)
                    new_players.append(second_child)
                return new_players

    def clone_player(self, player):
        """
        Gets a player as an input and produces a clone of that player.
        """
        new_player = Player(self.game_mode)
        new_player.nn = copy.deepcopy(player.nn)
        new_player.fitness = player.fitness
        return new_player

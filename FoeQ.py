import numpy as np
import matplotlib.pyplot as plt
import sys
from soccer import World1,Player
from cvxopt import matrix, solvers


#soccer env credits - https://github.com/pdvelez/ml_soccer
def init_env():
    env = World1()
    #4 cols, 2 rows
    env.set_world_size(4,2)
    # env.set_commentator_on()
    env.init_grid()
    env.set_goals(100,0,'0')
    env.set_goals(100,3,'1')
    # col, row, has_ball, id
    playerA = Player(2, 0, 0, '0')
    playerB = Player(1, 0, 1, '1')
    env.place_player(playerA, '0')
    env.place_player(playerB, '1')
    # env.plot_grid()
    return env

#test environment
def test_env():
    env = init_env()
    a = {'0': 2, '1': 2}
    env.move(a)
    env.plot_grid()
    a = {'0': 2, '1': 2}
    print(env.move(a))
    env.plot_grid()


class FoeQ_learning():
    def __init__(self):
        self.gamma = 0.9
        self.epsilon = 1.0
        self.alpha = 1.0
        self.epsilon_limit = 0.001
        self.alpha_limit = 0.001
        self.seed = 55
        self.QA = np.ones((8,8,2,5,5))
        self.QB = np.ones((8,8,2,5,5))
        self.PA = np.full((8,8,2,5),1/5)
        self.PB = np.full((8,8,2,5),1/5)
        self.VA = np.ones((8,8,2))
        self.VB = np.ones((8,8,2))
        self.step_errors = []
        self.no_eps = int(1e6)
        # Matrices for constraints
        self.A = matrix([[0.0],[1.0],[1.0],[1.0],[1.0],[1.0]]) #p coef for sum
        self.B = matrix(1.0)  #sum = 1
        self.C = matrix([-1.0,0.0,0.0,0.0,0.0,0.0]) #solution coef
        self.H = matrix(np.zeros((10,1))) #constraints<=0
        self.epsilon_decay = 10 ** (np.log10(self.epsilon_limit) / self.no_eps)
        self.alpha_decay = 10 ** (np.log10(self.alpha_limit) / self.no_eps)
        self.record_actions = []

    def get_action(self,state,player):
        greed = np.random.random()
        if greed<self.epsilon:
            return np.random.randint(0,5)
        else:
            if player:
                action = np.random.choice(5,1,p=self.PB[int(state[1])][int(state[2])][int(state[0])])
            else:
                action = np.random.choice(5,1,p=self.PA[int(state[1])][int(state[2])][int(state[0])])
        return action[0]

    def create_G(self,state,Qi):
        Reward_matrix = -Qi[int(state[1])][int(state[2])][int(state[0])]
        ones = np.ones((5,1))
        reward_constraints = np.hstack((ones,Reward_matrix))
        zeros = np.zeros((5, 1))
        I = np.eye(5)
        probability_constraints = np.hstack((zeros, -I))
        G = np.vstack((reward_constraints,probability_constraints))
        return matrix(G)

    def minimax_lp(self,state,Qi):
        self.G = self.create_G(state,Qi)
        solution = solvers.lp(self.C,self.G,self.H,self.A,self.B)
        return np.array(solution['x'])

    def solve(self):
        i = 0
        solvers.options['show_progress'] = False
        while i <self.no_eps:
            env = init_env()
            state = env.map_player_state() #ball,A,Ba
            while True:
                i+=1
                a1 = self.get_action(state,0)
                a2 = self.get_action(state,1)
                graph_val1 = self.QA[2][1][1][4][2]
                actions = {'0':a1,'1':a2}
                self.record_actions.append([actions['0'], actions['1']])
                new_state,rewards,done = env.move(actions)

                # Solve minimax using lp for player A
                solA = self.minimax_lp(new_state, self.QA)
                piA = solA[1:]
                piA = np.abs(piA).reshape((5,)) / sum(np.abs(piA)) #Normalize policy
                self.PA[int(new_state[1])][int(new_state[2])][int(new_state[0])] = piA
                self.VA[int(new_state[1])][int(new_state[2])][int(new_state[0])] = solA[0]

                # Solve minimax using lp for player B
                solB = self.minimax_lp(new_state, self.QB)
                piB = solB[1:]
                piB = np.abs(piB).reshape((5,)) / sum(np.abs(piB)) #Normalize policy
                self.PB[int(new_state[1])][int(new_state[2])][int(new_state[0])] = piB
                self.VB[int(new_state[1])][int(new_state[2])][int(new_state[0])] = solB[0]

                VA = self.VA[int(new_state[1])][int(new_state[2])][int(new_state[0])]
                self.QA[int(state[1])][int(state[2])][int(state[0])][actions['1']][actions['0']] = self.QA[int(state[1])][int(state[2])][int(state[0])][actions['1']][actions['0']] \
                + self.alpha * (rewards['0'] + self.gamma * VA - self.QA[int(state[1])][int(state[2])][int(state[0])][actions['1']][actions['0']])

                VB = self.VB[int(new_state[1])][int(new_state[2])][int(new_state[0])]
                self.QB[int(state[1])][int(state[2])][int(state[0])][actions['0']][actions['1']] = self.QB[int(state[1])][int(state[2])][int(state[0])][actions['0']][actions['1']] \
                + self.alpha * (rewards['1'] + self.gamma * VB - self.QB[int(state[1])][int(state[2])][int(state[0])][actions['0']][actions['1']])

                state = new_state
                graph_val2 = self.QA[2][1][1][4][2]
                self.step_errors.append(abs(graph_val2 - graph_val1))
                self.epsilon *= self.epsilon_decay
                self.alpha *= self.alpha_decay
                if done:
                    break

        return self.step_errors,self.record_actions


def fig3():
    solver = FoeQ_learning()
    errors, actions = solver.solve()
    errors = np.array(errors)
    actions = np.array(actions).T
    plt.plot(errors, linestyle='-', linewidth=0.3)
    plt.title("Fig 3.a Foe Q-Learning")
    plt.ylim(0, 0.75)
    plt.xlabel('Simulation Iteration')
    plt.ylabel('Q-value Difference')
    plt.show()
    #Fig actions
    p1_actions = []
    for i in range(5):
        p1_actions.append(np.count_nonzero(actions[0] == i))
    p2_actions = []
    for i in range(5):
        p2_actions.append(np.count_nonzero(actions[1] == i))
    barWidth = 0.35
    plt.subplots(figsize=(10, 8))
    br1 = np.arange(len(p1_actions))
    br2 = [x + barWidth for x in br1]
    plt.bar(br1, p1_actions, color='b', width=barWidth,
            edgecolor='grey', label='Player A')
    plt.bar(br2, p2_actions, color='r', width=barWidth,
            edgecolor='grey', label='Player B')
    plt.title("Fig 3.b Actions in Foe Q-learning")
    plt.xticks([r + barWidth for r in range(5)],
               ['North', 'South', 'East', 'West', 'Stick'])
    plt.xlabel("Actions")
    plt.ylabel("Number of actions in 1M")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    globals()[sys.argv[1]]()
















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


class uCEQ_learning():
    def __init__(self):
        self.gamma = 0.9
        self.epsilon = 1.0
        self.alpha = 1.0
        self.epsilon_limit = 0.001
        self.alpha_limit = 0.001
        self.seed = 55
        self.QA = np.ones((8,8,2,5,5))
        self.QB = np.ones((8,8,2,5,5))
        self.P = np.full((8,8,2,5,5),1/25)
        self.VA = np.ones((8,8,2))
        self.VB = np.ones((8,8,2))
        self.step_errors = []
        self.no_eps = int(1e6)
        # Matrices for constraints
        self.A = matrix(np.ones((1,25))) #p coef for sum
        self.B = matrix(1.0)  #sum = 1
        self.H = matrix(np.zeros(65)) #constraints<=0
        #decay schedule
        self.epsilon_decay = 10 ** (np.log10(self.epsilon_limit) / self.no_eps)
        self.alpha_decay = 10 ** (np.log10(self.alpha_limit) / self.no_eps)
        self.record_actions = []

    def get_actions(self,state):
        greed = np.random.random()
        if greed<self.epsilon:
            action = np.random.randint(0,25)
            return action//5,action%5
        else:
            action = np.random.choice(25,1,p=self.P[int(state[1])][int(state[2])][int(state[0])].reshape(25))
            return action[0]//5,action[0]%5

    def create_G(self,QA, QB):
        grid_1 = np.zeros((QA.shape[0] * QA.shape[0], QA.shape[1] * QA.shape[1]))
        for i in range(QA.shape[0]):
            for j in range(QA.shape[0]):
                row_index = (i * 5) + j
                a = QA[j, :] - QA[i, :]
                col_index = i * 5
                grid_1[row_index, col_index: col_index + 5] = a
        grid_2 = np.zeros((QB.shape[0] * QB.shape[0], QB.shape[1] * QB.shape[1]))
        for i in range(QB.shape[0]):
            for j in range(QB.shape[0]):
                row_index = (i * 5) + j
                a = QB[j, :] - QB[i, :]
                col_index = i * 5
                grid_2[row_index, col_index: col_index + 5] = a
        z = []
        for i in range(QA.shape[0]):
            z.append((i * 5) + i)
        parameters1 = np.delete(grid_1, z, axis=0)
        c = []
        for i in range(5):
            for j in range(5):
                c.append((j * 5) + i)
        parameters2 = np.delete(grid_2, z, axis=0)
        parameters2 = parameters2[:, c]
        grid_5 = np.vstack((parameters1, parameters2))
        probs = -np.eye(25)
        G = np.append(grid_5, probs, axis=0)
        return matrix(G)

    def ceq_lp(self,state):
        QA = self.QA[int(state[1])][int(state[2])][int(state[0])]
        QB = self.QB[int(state[1])][int(state[2])][int(state[0])]
        self.C = matrix((QA + QB.T).reshape(25))
        self.G = self.create_G(QA,QB)
        try:
            solution = solvers.lp(self.C, self.G, self.H, self.A, self.B)
            if solution is None:
                return None
            else:
                return np.array(solution['x'])
        except:
            pass

    def solve(self):
        i = 0
        solvers.options['show_progress'] = False
        while i <self.no_eps:
            env = init_env()
            state = env.map_player_state() #ball,A,Ba
            while True:
                i+=1
                a1,a2 = self.get_actions(state)
                graph_val1 = self.QA[2][1][1][2][4]
                actions = {'0':a1,'1':a2}
                self.record_actions.append([actions['0'], actions['1']])
                new_state,rewards,done = env.move(actions)

                # Solve ceq using lp
                solution = self.ceq_lp(new_state)
                if solution is not None:
                    pi = np.abs(solution.reshape((5,5))) / sum(np.abs(solution)) #Normalize policy
                    self.P[int(new_state[1])][int(new_state[2])][int(new_state[0])] = pi
                    self.VA[int(new_state[1])][int(new_state[2])][int(new_state[0])] = np.sum(pi * self.QA[int(new_state[1])][int(new_state[2])][int(new_state[0])])
                    self.VB[int(new_state[1])][int(new_state[2])][int(new_state[0])] = np.sum(pi * self.QB[int(new_state[1])][int(new_state[2])][int(new_state[0])].T)

                VA = self.VA[int(new_state[1])][int(new_state[2])][int(new_state[0])]
                self.QA[int(state[1])][int(state[2])][int(state[0])][actions['0']][actions['1']] = self.QA[int(state[1])][int(state[2])][int(state[0])][actions['0']][actions['1']] \
                + self.alpha * (rewards['0'] + self.gamma * VA - self.QA[int(state[1])][int(state[2])][int(state[0])][actions['0']][actions['1']])

                VB = self.VB[int(new_state[1])][int(new_state[2])][int(new_state[0])]
                self.QB[int(state[1])][int(state[2])][int(state[0])][actions['1']][actions['0']] = self.QB[int(state[1])][int(state[2])][int(state[0])][actions['1']][actions['0']] \
                + self.alpha * (rewards['1'] + self.gamma * VB - self.QB[int(state[1])][int(state[2])][int(state[0])][actions['1']][actions['0']])

                state = new_state
                graph_val2 = self.QA[2][1][1][2][4]
                self.step_errors.append(abs(graph_val2 - graph_val1))
                self.epsilon *= self.epsilon_decay
                self.alpha *= self.alpha_decay
                if done:
                    break

        return self.step_errors,self.record_actions


def fig2():
    solver = uCEQ_learning()
    errors, actions = solver.solve()
    errors = np.array(errors)
    actions = np.array(actions).T
    plt.plot(errors, linestyle='-', linewidth=0.4)
    plt.title("Fig 2.a uCE Q-Learning")
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
    plt.title("Fig 2.b Actions in uCE Q-learning")
    plt.xticks([r + barWidth for r in range(5)],
               ['North', 'South', 'East', 'West', 'Stick'])
    plt.xlabel("Actions")
    plt.ylabel("Number of actions in 1M")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    globals()[sys.argv[1]]()


















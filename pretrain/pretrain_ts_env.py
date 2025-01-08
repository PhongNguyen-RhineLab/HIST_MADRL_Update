import numpy as np
import sys
import random
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

ENV_H = 15  # env height
ENV_W = ENV_H  # env width
UNIT = 20  # grid size

class ENV(tk.Tk, object):
    def __init__(self, agentNum):
        super(ENV, self).__init__()
        self.ENV_H = 15  # Environment height
        self.UNIT = 20   # Grid size
        self.agentNum = agentNum
        self.n_actions = self.agentNum
        self.historyStep = 1
        self.n_features = 2 * (2 * self.agentNum - 1) + 2 * (self.agentNum - 1) * self.historyStep * 2
        self.agent_all = [None] * self.agentNum
        self.target_all = [None] * self.agentNum
        self.agentSize = 0.25 * self.UNIT
        self.tarSize = 0.5 * self.UNIT
        self.agent_center = np.zeros((self.agentNum, 2))
        self.tar_center = np.zeros((self.agentNum, 2))
        self.geometry(f'{self.ENV_H * self.UNIT}x{self.ENV_H * self.UNIT}')
        random_multiplier = random.uniform(0.5, 1)
        self.agent_speeds = self.agentNum * random_multiplier  # Random speeds for each agent
        self._build_env()


    def _build_env(self):
        self.canvas = tk.Canvas(self, bg='white',
                                height=self.ENV_H * 20,  # UNIT
                                width=self.ENV_H * 20)  # UNIT
        self.origin = np.array([20 / 2, 20 / 2])  # HalfUnit
        for i in range(self.agentNum):
            self.tar_center[i] = self.origin + 20 * np.random.rand(2) * (self.ENV_H - 1 + 0.01)
            self.agent_center[i] = self.origin + 20 * np.random.rand(2) * (self.ENV_H - 1 + 0.01)
            self.agent_all[i] = self.canvas.create_oval(
                self.agent_center[i, 0] - self.agentSize, self.agent_center[i, 1] - self.agentSize,
                self.agent_center[i, 0] + self.agentSize, self.agent_center[i, 1] + self.agentSize,
                fill='green')
            self.target_all[i] = self.canvas.create_rectangle(
                self.tar_center[i, 0] - self.tarSize, self.tar_center[i, 1] - self.tarSize,
                self.tar_center[i, 0] + self.tarSize, self.tar_center[i, 1] + self.tarSize,
                fill='orange')
        self.canvas.pack()

    def reset(self, agentPositionArray, tarPositionArray):
        self.update()
        self.agent_speeds = np.random.rand(self.agentNum)  # Reassign random speeds on reset
        sATAA = np.zeros((self.agentNum, 2 * (2 * self.agentNum - 1)))
        agent_coordi = np.zeros((self.agentNum, 2))
        tar_coordi = np.zeros((self.agentNum, 2))
        for i in range(self.agentNum):
            self.canvas.delete(self.agent_all[i])
            self.canvas.delete(self.target_all[i])
        self.agentPositionArray = agentPositionArray
        self.tarPositionArray = tarPositionArray
        for i in range(self.agentNum):
            self.tar_center[i] = self.origin + UNIT * self.tarPositionArray[i]
            self.agent_center[i] = self.origin + UNIT * self.agentPositionArray[i]
            self.target_all[i] = self.canvas.create_rectangle(
                self.tar_center[i, 0] - self.tarSize, self.tar_center[i, 1] - self.tarSize,
                self.tar_center[i, 0] + self.tarSize, self.tar_center[i, 1] + self.tarSize,
                fill='red')
            self.agent_all[i] = self.canvas.create_oval(
                self.agent_center[i, 0] - self.agentSize, self.agent_center[i, 1] - self.agentSize,
                self.agent_center[i, 0] + self.agentSize, self.agent_center[i, 1] + self.agentSize,
                fill='blue')
        for i in range(self.agentNum):
            tar_coordi[i] = np.array(self.canvas.coords(self.target_all[i])[:2]) + np.array([self.tarSize, self.tarSize])
            agent_coordi[i] = np.array(self.canvas.coords(self.agent_all[i])[:2]) + np.array([self.agentSize, self.agentSize])
        for i in range(self.agentNum):
            for k in range(self.agentNum):
                sATAA[i, 2*k: 2*(k+1)] = (tar_coordi[k] - agent_coordi[i])/(ENV_H * UNIT)
            for j in range(self.agentNum):
                if j > i:
                    sATAA[i, 2*(self.agentNum + j-1): 2*(self.agentNum + j)] = (agent_coordi[j] - agent_coordi[i])/(ENV_H * UNIT)
                elif j < i:
                    sATAA[i, 2*(self.agentNum + j): 2*(self.agentNum + j)+2] = - sATAA[j, 2*(self.agentNum + i-1): 2*(self.agentNum + i)]
        return sATAA

    def step(self, action, observation, agentiDone):
        base_actionA = np.array([0.0, 0.0])
        if agentiDone != action + 1:
            move_vector = observation[action * 2:(action + 1) * 2]
            speed = self.agent_speeds[action]
            norm = np.linalg.norm(move_vector)
            if norm != 0:
                base_actionA += (move_vector / norm) * speed * 20  # UNIT
        return base_actionA[0], base_actionA[1]

    def move(self, move, action):
        """
        Move agents based on their actions, calculate rewards, collisions, and update state.

        Parameters:
            move (ndarray): The movement vectors for agents.
            action (ndarray): The action indices for agents.

        Returns:
            sATAA (ndarray): State array after the move.
            reward (ndarray): Rewards for each agent.
            done (bool): Whether the episode has ended.
            agentDone (list): List indicating which target each agent has reached.
            collision (ndarray): Array indicating collision status for each agent.
            collision_true (int): Whether a collision has occurred between agents.
            success (int): Whether all agents have reached their targets successfully.
            conflict (int): Whether any conflict occurred between agents.
        """
        # Initialization
        collision = np.zeros(self.agentNum)
        reward = -1 * np.ones(self.agentNum) / self.ENV_H
        agentDone = [0] * self.agentNum  # Agent target reach status
        tarDone = [0] * self.agentNum  # Target reach status
        conflict_num = np.zeros(self.agentNum)  # Conflicts for each agent
        collision_true = 0
        success = 0
        conflict = 0

        # Calculate conflicts (agents choosing the same target)
        for i in range(self.agentNum):
            for k in range(self.agentNum):
                if k != i and action[i] == action[k]:
                    conflict_num[i] += 1

        # Move agents with speed scaling and update positions
        for i in range(self.agentNum):
            scaled_move = move[i] * self.agent_speeds[i]
            self.canvas.move(self.agent_all[i], scaled_move[0], scaled_move[1])

        # Calculate positions of agents and targets
        agent_coordi = np.zeros((self.agentNum, 2))
        tar_coordi = np.zeros((self.agentNum, 2))
        for i in range(self.agentNum):
            tar_coordi[i] = np.array(self.canvas.coords(self.target_all[i])[:2]) + [self.tarSize, self.tarSize]
            agent_coordi[i] = np.array(self.canvas.coords(self.agent_all[i])[:2]) + [self.agentSize, self.agentSize]

        # Sort agents by x-coordinate (for distance calculations)
        sortAgent_index = np.argsort(agent_coordi[:, 0])

        # Initialize state array
        sATAA = np.zeros((self.agentNum, 2 * (2 * self.agentNum - 1)))

        # Compute distances between agents and targets
        for i in range(self.agentNum):
            for k in range(self.agentNum):  # Distance between agents and targets
                sATAA[i, 2 * k: 2 * (k + 1)] = (tar_coordi[k] - agent_coordi[i]) / (self.ENV_H * self.UNIT)
            temp = 0
            for j in range(self.agentNum):  # Distance between agents
                if sortAgent_index[j] != i:
                    sATAA[i, 2 * (self.agentNum + temp): 2 * (self.agentNum + temp + 1)] = \
                        (agent_coordi[sortAgent_index[j]] - agent_coordi[i]) / (self.ENV_H * self.UNIT)
                    temp += 1

        # Calculate collisions between agents
        agent_pair_index = 0
        for i in range(self.agentNum):
            for k in range(self.agentNum):
                if k > i:  # Avoid double-checking pairs
                    distance = np.linalg.norm(agent_coordi[i] - agent_coordi[k])
                    if distance < self.UNIT:  # Collision detected
                        collision[i], collision[k] = 1, 1
                        if action[i] == action[k]:
                            collision_true = 1
                    agent_pair_index += 1

        # Assign rewards based on collisions and conflicts
        for i in range(self.agentNum):
            reward[i] += -30 * conflict_num[i] / self.ENV_H
            for j in range(self.agentNum):
                if j > i:
                    if agentDone[i] == agentDone[j] and agentDone[i] != 0:
                        if action[i] == action[j]:
                            conflict = 1
                            reward[i] += -45 / self.ENV_H
                            reward[j] += -45 / self.ENV_H
                            break

        # Reward for reaching targets
        if np.sum(tarDone) == self.agentNum:
            reward += 0.8 * np.ones(self.agentNum) / self.ENV_H

        # Check for episode completion
        if np.sum(tarDone) == self.agentNum:
            done = True
            success = 1
        elif conflict == 1 or collision_true == 1:
            done = True
        else:
            done = False

        return sATAA, reward, done, agentDone, collision, collision_true, success, conflict

    def render(self):
        self.update()
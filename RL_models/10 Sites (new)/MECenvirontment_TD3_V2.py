#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 15:34:54 2021

@author: widhi
"""
import random
from typing import List
import numpy as np
import math
from  Propagation import Propagation
from sklearn import preprocessing

huge = 100

class EnvironmentTD3:
    def __init__(self):
        # Basic environtment setup
        self.steps_left = 10
        self.total_reward = 0.0
        self.done = False
        
        # topology settings
        self.BUA = 1000000
        self.BAU = 1000000
        self.CNNum = 1
        self.ANNum = 9
        self.propagationCN = Propagation("cnmec3.csv", self.CNNum).propagation_delay
        self.propagationAN = Propagation("anmec.csv", self.ANNum).propagation_delay
        
        # Initial capacity for each tiers and sites
        InitmiuCn = [251000 for i in range(self.CNNum)]
        InitmiuAn = [[30000 for i in range(self.ANNum)] for j in range(self.CNNum)]
        
        # Arrival traffic rates
        self.lam = (2000,4000,2000,4000,2000,4000,2000,4000,2000)
        np.random.seed(0)
        self.lamda = np.random.poisson(lam=self.lam, size=(self.CNNum, self.ANNum))
        
        # Array to collect latency (initial will 0) updated during calculation
        self.latency = [np.zeros((self.ANNum,self.ANNum+self.CNNum)) for i in range(self.CNNum)]

        self.AN = [[0 for i in range(self.ANNum)] for j in range(self.CNNum)]
        self.CN = [0 for i in range(self.CNNum)]
        
    def get_observation(self) :
        latency = self.latency
        lamda = self.lamda
        BUA = self.BUA
        BAU = self.BAU
        self.InitmiuCn = [260000 for i in range(self.CNNum)]
        self.InitmiuAn = [[30000 for i in range(self.ANNum)] for j in range(self.CNNum)]
        miuAN = self.InitmiuAn
        for i in range(len(miuAN)):
            for j in range(len(miuAN[i])):
                miuAN[i][j]-= self.AN[i][j] 
        miuCN = self.InitmiuCn
        for m in range(len(miuCN)):
            miuCN[m] -= self.CN[m]
        # print("AN miuAN===================== ", miuAN)
        # print("CN miuCN=====================", miuCN)
        scaler = preprocessing.RobustScaler()
        ob = [latency, lamda, BUA, BAU, miuAN, miuCN]
        l = scaler.fit_transform(np.array(latency).flatten().reshape(-1,1))
        a = scaler.fit_transform(np.array(lamda).flatten().reshape(-1,1))
        # b = scaler.fit_transform(np.array([BUA,BAU]).reshape(-1,1))
        man = scaler.fit_transform(np.array(miuAN).flatten().reshape(-1,1))
        mac = scaler.fit_transform(np.array(miuCN).flatten().reshape(-1,1))
       
        # halo = [val for sublist in latency for val in sublist]
        # print(l)
        # out = np.concatenate(halo)
        observe = np.concatenate((l,a,man,mac)).flatten()
        # print(observe)
        return ob,l,observe 
        
    
    def get_actions(self):
        
        self.InitmiuCn = [260000 for i in range(self.CNNum)]
        self.InitmiuAn = [[30000 for i in range(self.ANNum)] for j in range(self.CNNum)]
        miuAN = self.InitmiuAn
        for i in range(len(miuAN)):
            for j in range(len(miuAN[i])):
                miuAN[i][j]-= self.AN[i][j] 
        miuCN = self.InitmiuCn
        for m in range(len(miuCN)):
            miuCN[m] -= self.CN[m]

        act = [np.zeros((self.ANNum, self.CNNum+self.ANNum)) for i in range(self.CNNum)]
        # print("act=1 ", act)
        # act2 = miuAN
        for i in range(self.CNNum):
            for j in range(self.ANNum):
                tmp = np.concatenate((np.array(miuAN[i]).flatten(), np.array(miuCN).flatten()))
                for k in range(len(tmp)):
                    act[i][j][k] = tmp[k]/sum(tmp)                    
        # print("act= ", np.array(act).flatten())

        return act, np.array(act).flatten()
    
    def traffic_alloc(self, b):
        # print("len b==", len(b))
        lamdax=[]
        for n in range(self.CNNum):
            lamdax.append(np.random.rand(self.ANNum,self.ANNum+self.CNNum))
        
        for i in range(len(lamdax)):
            for j in range(len(lamdax[i])):
                for k in range(len(lamdax[i][j])):
                    lamdax[i][j][k] = math.ceil(self.lamda[i][j]*round(b[i][j][k],2))
            
        x=[]            
        for m in range(len(lamdax)):
            x.append(lamdax[m].sum(axis=1))
        
        for i in range(len(x)):
            for j in range(len(x[i])):
                if x[i][j] < self.lamda[i][j]:
                    lamdax[i][j][random.randint(0,len(lamdax[i][j])-1)] += self.lamda[i][j] - x[i][j]
                elif x[i][j] > self.lamda[i][j]:
                    lamdax[i][j][random.randint(0,len(lamdax[i][j])-1)] -= x[i][j] - self.lamda[i][j]
                else:
                    continue
        
        return lamdax
                
    def get_latency(self, miu, lamda):
        lat = 1/(miu-lamda)
        return lat
    
    
    def get_latency_local_AN(self, BUA, BAU, miuA, lamdaA, lamda):
        if (miuA - lamdaA) <= 0:
            latency = -(miuA - lamdaA)
        else:
            latency = 1/(BUA-lamda) + 1/(miuA-lamdaA) + 1/(BAU - lamda)
        return latency
    
    
    def get_latency_neighbour_AN(self, BUA, BAU, miuA, lamdaA, lamda, DAA):
        if (miuA - lamdaA) <=0:
            latency = -(miuA - lamdaA)
        else:    
            latency = 1/(BUA-lamda) + 1/(miuA-lamdaA) + 1/(BAU - lamda) + 2*DAA
        return latency
    
    def get_latency_CN(self, BUA, BAU, miuC, lamdaC, lamda, DAC):
        if (miuC-lamdaC) <= 0:
            latency = -(miuC-lamdaC)
        else:
            latency = 1/(BUA-lamda) + 1/(miuC-lamdaC) + 1/(BAU - lamda) + 2*DAC
        return latency
    
    def get_latency_neighbour_CN(self, BUA, BAU, miuC, lamdaC, lamda, DAC, DCC):
        if (miuC-lamdaC) <= 0:
            latency = -(miuC-lamdaC)
        else:
            latency = 1/(BUA-lamda) + 1/(miuC-lamdaC) + 1/(BAU - lamda) + 2*DAC +2*DCC
        return latency
    
    def get_lamda_AN(self, i, k, actions):
        lamdaAN = 0
        for j in range(self.ANNum):
            lamdaAN += actions[i][j][k]

        return lamdaAN
    
    def get_lamda_CN(self, k, actions):
        lamdaCN = 0
        for i in range(self.CNNum):
            for j in range(self.ANNum):
                # print("CN :", actions[i][j][k])
                lamdaCN += actions[i][j][k]
        
        return lamdaCN
    
    def is_done(self, step):
        if step == 0:
            self.done = True
        else:
            self.done = False
        return self.done
    
    # Calculate reward for given actions
    def action(self, obs, actions):
        latency = obs[0]
        lamda = obs[1]
        BUA = obs[2]
        BAU = obs[3]
        miuAN = obs[4]
        miuCN = obs[5]
        b = actions
        actions = self.traffic_alloc(b)
        # print("latency: ", latency)
        for i in range(len(latency)):
            lamdaAN =0
            for j in range(len(latency[i])):
                
                for k in range(len(latency[i][j])):
                    lamdaAN = self.get_lamda_AN(i,k, actions)
                    lamdaCN = self.get_lamda_CN(k, actions)
                    
                    ## Latency that is served at local AN
                    if j==k and k<=self.ANNum:
                        latency[i][j][k] =self.get_latency_local_AN(BUA, BAU, miuAN[i][j], lamdaAN, lamda[i][j]) 
                        
                    ## Latency that is served at AN Neighbor
                    elif k < self.ANNum and k!=j:
                        DAA = self.propagationAN()[j][k]
                        # print ("propagationAN : ", self.propagationAN())
                        latency[i][j][k] = self.get_latency_neighbour_AN(BUA, BAU, miuAN[i][j], lamdaAN, lamda[i][j], DAA)
                        
                    ## Latency that is served at local CN
                    elif self.ANNum-1 < k < (self.ANNum + self.CNNum) and k == self.ANNum+i:
                        latency[i][j][k] = self.get_latency_CN(BUA,BAU, miuCN[i], lamdaCN, lamda[i][j], 0.0001)
                        
                    ## Latency that is served at CN neighbour
                    elif self.ANNum-1 < k < (self.ANNum + self.CNNum) and k != self.ANNum+i:
                        # print(k)
                        DCC = self.propagationCN()[i][k-self.ANNum]
                        # print ("DCC : ", DCC)
                        latency[i][j][k] = self.get_latency_neighbour_CN(BUA,BAU, miuCN[i], lamdaCN, lamda[i][j], 0.0001, DCC)
        
        
        total_latency = 0
        for m in range(len(latency)):
            for n in range(len(latency[m])):
                for o in range(len(latency[m][n])):
                    total_latency += (actions[m][n][o]*latency[m][n][o])
                 
        total_traffic = sum(sum(map(sum, actions)))
        
        avg_delay = total_latency/total_traffic
        
        reward = 1/avg_delay
        if self.done == True:
            self.reset()
            self.AN = [[0 for i in range(self.ANNum)] for j in range(self.CNNum)]
            self.CN = [0 for i in range(self.CNNum)]
            pass
            # raise Exception("An episode is done")
            
        # self.steps_left -= 1
        return reward
    
    def step(self, actions):
        curret_obs, obs2, obs = self.get_observation()
        # print ("Current obs : ", curret_obs)
        reward = self.action(curret_obs, actions)
        self.total_reward += reward
        actions = self.traffic_alloc(actions)
        # print ("############################")
        # print ("reward: ", self.total_reward)
        # print ("actions:", actions)
        A = [] 
        for i in actions:
            A.append([sum(x) for x in zip(*i)])
        # print ("AAAAAAAAAAA=========================>",A)
        B = [sum(x) for x in zip(*A)]
        # print ("BBBBBBBBBBBB=========================>",B)
        for i in range(len(self.CN)):
            self.CN[i] += B[self.ANNum+i]
            for j in range(len(self.AN[i])):
                self.AN[i][j] += A[i][j]
        curret_obs, obs2, obs = self.get_observation()
        self.steps_left -= 1
        done = self.is_done(self.steps_left)
        # print ("after:", obs)
        # print ("############################")
        return obs, reward, done
    
    
    def reset(self):
        self.steps_left = 10
        self.AN = [[0 for i in range(self.ANNum)] for j in range(self.CNNum)]
        self.CN = [0 for i in range(self.CNNum)]
        self.latency = [np.zeros((self.ANNum,self.ANNum+self.CNNum)) for i in range(self.CNNum)]
        self.lamda = np.random.poisson(lam=self.lam, size=(self.CNNum, self.ANNum))
        self.BUA = 1000000
        self.BAU = 1000000
        InitmiuCn = [260000 for i in range(self.CNNum)]
        InitmiuAn = [[30000 for i in range(self.ANNum)] for j in range(self.CNNum)]
        # env.is_done = False
        
if __name__ == "__main__":
    
    ############################ Main Step #####################################################
    env = EnvironmentTD3()
    # agent = Agent()
    # actions = env.get_actions()
    import timeit  
    done = False
    rew = 0
    while not done:
        # print("Action : ", env.get_actions())
        start = timeit.default_timer()
        action, array = env.get_actions()
        # print("action: ", action)
        stop = timeit.default_timer()
        execution_time = stop - start
        print("Program Executed in "+str(execution_time))
        obs, obs2, obs3= env.get_observation()
        # print("obs", obs)
        # print("obs2", obs2)
        # print("obs3", obs3)
        obs, reward, done = env.step(action)
        rew += reward
        print("reward: ", reward)
        print("AN after : ", env.AN)
        print("CN after : ", env.CN)
        # print("Total: ", sum(map(sum, env.AN))+sum(env.CN))
        # print("##################################################")
        # print("done=", done)
        if done == True:
            env.reset()
            # print("AN afyer : ",env.AN)
            obs, obs2, obs3 = env.get_observation()
            # print("obs32====>", obs)
            # print("env step 2: ", env.steps_left)
            print("Total reward got : ", rew/10)
    
    ########################################################################################

    # print(env.get_actions())
    # print("AN before : ", env.AN)
    # print("CN before : ", env.CN)
    # print("======================")
    # print("miuAn====>", env.InitmiuAn)
    # print("miuCn====>", env.InitmiuCn)
    # print("reward=", reward)
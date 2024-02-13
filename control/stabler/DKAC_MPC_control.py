import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import gym
import matplotlib.pyplot as plt
import random
import argparse
from collections import OrderedDict
from copy import copy
import scipy
import scipy.linalg
import sys
sys.path.append("control/utility/")
# sys.path.append("control/utility/")
# sys.path.append("control/train/")
from Utility import data_collecter
import mpc
#import lqr
import os
from learnSOC_control import learnSOCmodel_withControl

Methods = ["KoopmanDerivative","KoopmanRBF",\
            "KNonlinear","KNonlinearRNN","KoopmanU",\
            "KoopmanNonlinearA","KoopmanNonlinear",\
                ]
method_index = 5

suffix = "compare_DKAC_sizeNN_30"
env_name = "CartPole-v1"
#env_name = "MountainCarContinuous-v0" 
# env_name = "DampingPendulum"
# env_name = "Pendulum-v1"

method = Methods[method_index]
root_path = "results/SOC_compare_sizeNNdata/"+suffix
print(method)
if method.endswith("KoopmanNonlinear"):
    import learn_DKN_SOC as lka
elif method.endswith("KoopmanNonlinearA"):
    import learn_DKAC_SOC as lka
elif method.endswith("KoopmanU"):
    import learn_DKUC_SOC as lka
for file in os.listdir(root_path):
    if file.startswith(method+"_"+env_name+"layer4") and file.endswith(".pth"):
        model_path = file  
Data_collect = data_collecter(env_name)
udim = Data_collect.udim
Nstate = Data_collect.Nstates
layer_depth = 4
layer_width = 128
dicts = torch.load(root_path+"/"+model_path,map_location=torch.device('cpu'))
state_dict = dicts["model"]
if method.endswith("KNonlinear"):
    Elayer = dicts["Elayer"]
    net = lka.Network(layers=Elayer,u_dim=udim)
elif method.endswith("KNonlinearRNN"):
    net = lka.Network(input_size=udim+Nstate,output_size=Nstate,hidden_dim=layer_width, n_layers=layer_depth-1)
elif method.endswith("KoopmanNonlinear") or method.endswith("KoopmanNonlinearA"):
    layer = dicts["layer"]
    blayer = dicts["blayer"]
    #elayer = dicts["elayer"]
    #dlayer = dicts["dlayer"]
    #A = dicts["LA"]
    #B = dicts["LB"]
    NKoopman = layer[-1]+Nstate
    net = lka.Network(layer,blayer,NKoopman,udim)#,elayer,dlayer
elif method.endswith("KoopmanU"):
    layer = dicts["layer"]
    NKoopman = layer[-1]+Nstate
    net = lka.Network(layer,NKoopman,udim)  
net.load_state_dict(state_dict)
print(NKoopman)
device = torch.device("cpu")
net.cpu()
net.double()

def Psi_o(s,net): # Evaluates basis functions Ψ(s(t_k))
    psi = np.zeros([NKoopman,1])
    ds = net.encode(torch.DoubleTensor(s)).detach().cpu().numpy()
    psi[:NKoopman,0] = ds
    return psi

def Prepare_MPC(env_name):#,Q0,Q1,Q2,Q3,R
    x_ref = np.zeros(Nstate)
    if env_name.startswith("CartPole"):
        Q = np.zeros((NKoopman,NKoopman))
        Q[0,0] = 0
        Q[1,1] = 0.01
        Q[2,2] = 5
        Q[3,3] = 0.01
        R = 0.05*np.eye(1) #0.8
        #reset_state = [0.0,0.0,-1.0,0.1]
        reset_state = [0.0,0.0,-1.0,0.1]
        #x_ref[0] = 1
        F = np.zeros((NKoopman,NKoopman))
        F[0,0] = Q[0,0]
        F[1,1] = Q[1,1]
        F[2,2] = Q[2,2]
        F[3,3] = Q[3,3]
        N_mpc = 20
    elif env_name.startswith("Pendulum"):
        Q = np.zeros((NKoopman,NKoopman))
        Q[0,0] = 5
        Q[1,1] = 0.01
        R = 500*np.eye(1) #0.01
        F = np.zeros((NKoopman,NKoopman))
        F[0,0] = Q[0,0]
        F[1,1] = Q[1,1]
        N_mpc = 10
        reset_state = [-3.0,6.0]
    elif env_name.startswith("DampingPendulum"):
        Q = np.zeros((NKoopman,NKoopman))
        Q[0,0] = 5
        Q[1,1] = 0.01
        R = 10*np.eye(1)
        reset_state = [-3.0,2.0]
        #x_ref[0] = 1
        F = np.zeros((NKoopman,NKoopman))
        F[0,0] = Q[0,0]
        F[1,1] = Q[1,1]
        N_mpc = 20   
    elif env_name.startswith("MountainCarContinuous"):
        Q = np.zeros((NKoopman,NKoopman))
        Q[0,0] = 5
        Q[1,1] = 0.01
        F = np.zeros((NKoopman,NKoopman))
        F[0,0] = Q[0,0]
        F[1,1] = Q[1,1]
        N_mpc = 20 
        R = 1*np.eye(1) #0.01
        #reset_state = [-0.3,0.1]
        reset_state = [-0.3,0.1]  
        x_ref[0] = 0.45
    Q = np.matrix(Q)
    R = np.matrix(R)
    F = np.matrix(F)
    return Q,R,F,reset_state,x_ref,N_mpc

def criterion(env_name,observations):
    if env_name.startswith("CartPole"):
        err = np.mean(abs(observations[2:,195:]))
    elif env_name.startswith("Pendulum"):
        err = np.mean(abs(observations[:,195:]))
    elif env_name.startswith("DampingPendulum"):
        print(observations.shape)
        err = np.mean(abs(observations[:,195:]))
    elif env_name.startswith("MountainCarContinuous"):
        err = np.mean(abs(observations[0,195:]-0.45))+np.mean(abs(observations[1,195:]))
    return err

def Cost(observations,u_list,Q,R,x_ref):
    steps = observations.shape[1]
    loss = 0
    for s in range(steps):
        if s!=steps-1:
            ucost = np.dot(np.dot(u_list[s].T,R),u_list[s])
            loss += ucost[0,0]
        xcost = np.dot(np.dot((observations[:,s]-x_ref).T,Q),(observations[:,s]-x_ref))
        loss += xcost[0,0]
    return loss

Ad = state_dict['lA.weight'].cpu().numpy()
Bd = state_dict['lB.weight'].cpu().numpy()
env = Data_collect.env
env.reset()
import lqr
import time
import mpc
from cvxopt import solvers
from cvxopt import matrix
Ad = np.matrix(Ad)#A_SOC
Bd = np.matrix(Bd)
Q,R,F,reset_state,x_ref,N_MPC = Prepare_MPC(env_name)
# reset_state=  [0.0,0.0,-1.0,0.1]
uval = 1.0
#Kopt = lqr.lqr_regulator_k(Ad,Bd,Q,uval*R,F,N_mpc)
M_MPC,C_MPC = mpc.cal_matrices(Ad,Bd,Q,uval*R,F,N_MPC)
M_MPC = matrix(M_MPC)

#print(Kopt.shape)
observation_list = []
observation = env.reset_state(reset_state)
x0 = np.matrix(Psi_o(observation,net))
x_ref_lift = Psi_o(x_ref,net)
#observation_list.append(x0[:Nstate].reshape(-1,1))
# print(Kopt)
#u_list = []
steps = 200
# umax = 100
x_observation = x0[:Nstate].reshape(-1,1)
Nx_observation = x_observation.shape[0]
#observation_list = np.zeros((Nx_observation, steps))
observation_list.append(x0[:Nstate].reshape(-1,1))
u_list = np.zeros((Bd.shape[1], steps))

for i in range(1,steps):
    # env.render()
    #u = -Kopt*(x0-x_ref_lift)
    x_kshort = x0-x_ref_lift
    x_kshort = x_kshort.reshape(-1,1)
    nx = x_kshort.shape[0]
    #x_kshort = X_k[:, k - 1].reshape(2, 1)
    u_kshort = u_list[:, i - 1].reshape(-1, 1)
    nu = u_kshort.shape[0]
    #print(x_kshort.shape)
    #print(u_kshort.shape)
    T_MPC = np.dot(C_MPC,x_kshort)
    T_MPC = matrix(T_MPC)
    #for k in range(nu):
    #print(M_MPC,T_MPC)
    u_list[Bd.shape[1]-1,i-1] = mpc.Prediction(M_MPC,T_MPC)#[k,0]
    gu = net.bilinear_net(torch.DoubleTensor(x0[:Nstate].reshape(1,-1))).detach().numpy()#[:Nstate]
    ureal = u_list[0,i-1]/gu[0,0]
    # u = max(-umax,min(umax,u[0,0]))
    # print(type(u[0,0]),type(u))
    # print(u_list[0,i-1])
    # print(gu[0,0])
    # print(ureal)
    observation, reward, done, info= env.step(ureal)
    # print(observation.shape)
    # X_knew = np.dot(Ad,x_kshort) + np.dot(Bd,u_kshort)
    x0 = np.matrix(Psi_o(observation,net))
    # x0 = Ad*x0+Bd*u
    # print(x0)
    observation_list.append(x0[:Nstate].reshape(-1,1))
    u_list[:, i] = ureal
    # time.sleep(0.1)

#print(observation_list)
observations = np.concatenate(observation_list,axis=1)
u_list = np.array(u_list).reshape(-1)
np.save("D:/毕业设计/中期/Python/MPC_trykoopman/control/stabler/outcomes/DKAC/"+env_name+"_"+method+"_obs.npy",observations)
Err = criterion(env_name,observations)
loss = Cost(observations,u_list,Q[:Nstate,:Nstate],0.001*R,x_ref)

print(Err,loss)
#print(env.dt)
time_history = np.arange(steps)*env.dt
""" plt.plot(time_history, u_list, label="u")
plt.show()  """
""" for i in range(Nstate):
    print(observations[i,-1]) """
for i in range(Nstate):
    print(observations[i,-1])
    plt.plot(time_history, observations[i,:].reshape(-1,1), label="x{}".format(i))
plt.grid(True)
plt.title("MPC_DKAC_SOC Regulator")
plt.legend()
#plt.savefig("D:/毕业设计/中期/Python/MPC_trykoopman/control/stabler/outcomes/pictures/"+env_name+"_"+method+".png",dpi = 400)#\yibantupian
plt.show() 
pass
    

# import numpy as np
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# mpl.rcParams['font.family'] = ['Times New Roman']
# mpl.rcParams["axes.titlepad"] = 16
# mpl.rcParams['lines.linewidth'] = 1
# mpl.rcParams['lines.markersize'] = 3
# mpl.rcParams['lines.markeredgecolor'] = 'black'
# mpl.rcParams['lines.markeredgewidth'] = '0.1'
# mpl.rcParams['xtick.bottom'] = False
# mpl.rcParams['ytick.left'] = False
# #plt.rcParams['figure.figsize'] = 4,3 #窗口大小
# plt.rcParams['figure.subplot.left'] = 0.05
# plt.rcParams['figure.subplot.right'] = 0.95
# plt.rcParams['figure.subplot.bottom'] = 0.3
# plt.rcParams['figure.subplot.top'] = 0.9 #子视图大小占视图的比例
# plt.rcParams['savefig.dpi'] = 500 #图片像素
# #plt.rcParams['figure.dpi'] = 300 #分辨率
# fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 3))
# #print(observation_list)
# # observations = np.concatenate(observation_list,axis=1)
# # u_list = np.array(u_list).reshape(-1)
# envnames = ["Pendulum-v1","MountainCarContinuous-v0","CartPole-v1","DampingPendulum"]
# labels = ["theta","dtheta"]
# title_names = ["(a) Pendulum","(b) MountainCarContinuous","(c) CartPole","(d)"]
# for p in range(3):
#     env_name = envnames[p]
#     title_name = title_names[p]
#     observations = np.load("D:/毕业设计/中期/Python/MPC_trykoopman/control/stabler/outcomes/DKAC/"+env_name+"_"+method+"_obs.npy")
#     time_history = np.arange(steps)*env.dt
#     if env_name=="CartPole-v1":
#         Nstate = 4
#         for i in range(2,Nstate):
#             print(observations[i,-1])
#             ax[p].plot(time_history, observations[i,:].reshape(-1,1), label=labels[i-2])
#     else:
#         Nstate = 2
#         for i in range(Nstate):
#             print(observations[i,-1])
#             ax[p].plot(time_history, observations[i,:].reshape(-1,1), label=labels[i])
#     ax[p].set_title(title_name,x=0.5,y=-0.4,fontsize=16)
#     #ax[p].legend()
#     ax[p].grid(True)
# plt.grid(True)
# plt.suptitle("SDKN-DKAC-MPC",x=0.5,y=1.0,fontsize=14)
# plt.legend(bbox_to_anchor=(-0.7, -0.3),loc='upper center', ncol=2,fontsize=12)#, mode='expand')
# plt.savefig("D:\毕业设计\论文\pictures/中期论文/SOC_"+method+"_draw_zhijeijieguo.png")#\yibantupian
# plt.show() 
# pass
    
# import numpy as np
# import matplotlib as mpl
# import matplotlib.pyplot as plt
# from scipy.interpolate import make_interp_spline
# from scipy.ndimage import gaussian_filter1d
# mpl.rcParams['font.family'] = ['Times New Roman']
# mpl.rcParams["axes.titlepad"] = 16
# mpl.rcParams['lines.linewidth'] = 1
# mpl.rcParams['lines.markersize'] = 3
# mpl.rcParams['lines.markeredgecolor'] = 'black'
# mpl.rcParams['lines.markeredgewidth'] = '0.1'
# mpl.rcParams['xtick.bottom'] = False
# mpl.rcParams['ytick.left'] = False
# #plt.rcParams['figure.figsize'] = 4,3 #窗口大小
# plt.rcParams['figure.subplot.left'] = 0.05
# plt.rcParams['figure.subplot.right'] = 0.95
# plt.rcParams['figure.subplot.bottom'] = 0.3
# plt.rcParams['figure.subplot.top'] = 0.95 #子视图大小占视图的比例
# plt.rcParams['savefig.dpi'] = 500 #图片像素
# #plt.rcParams['figure.dpi'] = 300 #分辨率
# fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 3))
# #print(observation_list)
# # observations = np.concatenate(observation_list,axis=1)
# # u_list = np.array(u_list).reshape(-1)
# envnames = ["Pendulum-v1","MountainCarContinuous-v0","CartPole-v1"]
# labels = ["theta","dtheta"]
# title_names = ["(a) Pendulum","(b) MountainCarContinuous","(c) CartPole"]
# methods = ["KoopmanU",\
#             "KoopmanNonlinearA","KoopmanNonlinear",\
#                 ]
# for p in range(3):
#     env_name = envnames[p]
#     title_name = title_names[p]
#     observations_A = np.load("D:/毕业设计/中期/Python/MPC_trykoopman/control/stabler/outcomes/DKAC/"+env_name+"_"+methods[1]+"_obs.npy")
#     observations_U = np.load("D:/毕业设计/中期/Python/MPC_trykoopman/control/stabler/outcomes/DKUC/"+env_name+"_"+methods[0]+"_obs.npy")
#     observations_N = np.load("D:/毕业设计/中期/Python/MPC_trykoopman/control/stabler/outcomes/DKN/"+env_name+"_"+methods[2]+"_obs.npy")
#     time_history = np.arange(steps)*env.dt
#     if env_name=="CartPole-v1":
#         Nstate = 4
#         # mA = make_interp_spline(observations_A[2,:].reshape(-1,1), observations_A[3,:].reshape(-1,1))
#         # xsA = np.linspace(min(observations_A[2,:].reshape(-1,1)), max(observations_A[2,:].reshape(-1,1)), 500)
#         # ysA = mA(xsA)
#         ax[p].plot(observations_A[2,:].reshape(-1,1), observations_A[3,:].reshape(-1,1),linestyle='-', label="SDKN-DKAC-MPC")
#         ax[p].plot(observations_U[2,:].reshape(-1,1), observations_U[3,:].reshape(-1,1),linestyle='-', label="SDKN-DKUC-MPC")
#         ax[p].plot(observations_N[2,:].reshape(-1,1), observations_N[3,:].reshape(-1,1),linestyle='-', label="SDKN-DKN-MPC")
#         ax[p].scatter([0], [0], color='red',marker='*', s=70)
#         ax[p].scatter([-1], [0.1], color='blue',marker='o')

#     elif env_name=="MountainCarContinuous-v0":
#         Nstate = 2
#         #for i in range(Nstate):
#         #print(observations[i,-1])
#         ax[p].plot(observations_A[0,:].reshape(-1,1), observations_A[1,:].reshape(-1,1),linestyle='-', label="SDKN-DKAC-MPC")
#         ax[p].plot(observations_U[0,:].reshape(-1,1), observations_U[1,:].reshape(-1,1),linestyle='-', label="SDKN-DKUC-MPC")
#         ax[p].plot(observations_N[0,:].reshape(-1,1), observations_N[1,:].reshape(-1,1), linestyle='-',label="SDKN-DKN-MPC")
#         ax[p].scatter([0.45], [0], color='red',marker='*', s=70)
#         ax[p].scatter([-0.3], [0.1], color='blue',marker='o')
#     elif env_name=="Pendulum-v1":
#         Nstate = 2
#         #for i in range(Nstate):
#         #print(observations[i,-1])
#         ax[p].plot(observations_A[0,:].reshape(-1,1), observations_A[1,:].reshape(-1,1),linestyle='-', label="SDKN-DKAC-MPC")
#         ax[p].plot(observations_U[0,:].reshape(-1,1), observations_U[1,:].reshape(-1,1),linestyle='-', label="SDKN-DKUC-MPC")
#         ax[p].plot(observations_N[0,:].reshape(-1,1), observations_N[1,:].reshape(-1,1), linestyle='-',label="SDKN-DKN-MPC")
#         ax[p].scatter([0], [0], color='red',marker='*', s=70)
#         ax[p].scatter([-3], [6], color='blue',marker='o')
#     ax[p].set_title(title_name,x=0.5,y=-0.4,fontsize=16)
#     #ax[p].legend()
#     ax[p].grid(True)
# plt.grid(True)
# #plt.suptitle("SDKN-DKAC-MPC",x=0.5,y=0.8,fontsize=14)
# plt.legend(bbox_to_anchor=(-0.7, -0.3),loc='upper center', ncol=3,fontsize=10)#, mode='expand')
# plt.savefig("D:\毕业设计\论文\pictures/中期论文/SOC_"+method+"_Phase_Diagram.png")#\yibantupian
# plt.show() 
# pass


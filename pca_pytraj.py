import pytraj as pt
import matplotlib.pyplot as plt
from pytraj.all_actions import projection
import numpy as np
import sys
import matplotlib as mpl
import os

# style stuff for plotting
mpl.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2
params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)

# traj = pt.load('oxa24_fullprodby10.nc','oxa24-WT.parm7') ### trajectory (.nc), parameters (.parm7)
# data = pt.pca(traj, mask='@CA', n_vecs = 20)

# # print('projection values of each frame to first mode = {} \n'.format(data[0][0]))
# # print('projection values of each frame to second mode = {} \n'.format(data[0][1]))
# # print('projection values of each frame to second mode = {} \n'.format(data[0][3]))
# # print('eigvenvalues', data[1][0])

# # projection = data[0]

# eigvals = data[1][0]
# eigvecs = data[1][1]

# # ### Stuff for plotting scree plot and PC projections
# # # plt.plot(range(10),data[1][0],marker = 'o')
# # # plt.title('Scree Plot of oxa24 WT')
# # # plt.xlabel('eigenvector')
# # # plt.ylabel('eigenvalue')
# # plt.scatter(projection[0],projection[1],marker='o',c=range(traj.n_frames),alpha=0.5)
# # plt.xlabel('PC1',fontsize = 24)
# # plt.ylabel('PC2',fontsize = 24)
# # plt.title('oxa24_R261S_fullprodby10 PCA score plot', fontsize = 24) ### rename title
# # cbar = plt.colorbar()
# # cbar.set_label('frame', fontsize = 16)
# # plt.show()

# ## Square Fluctionations for residue induce by the kth principal component
# def fluct(k,i): 
#     return (1/eigvals[k-1])*(np.dot(eigvecs[k-1][i-1],eigvecs[k-1][i-1]) + np.dot(eigvecs[k-1][i],eigvecs[k-1][i]) + np.dot(eigvecs[k-1][i+1],eigvecs[k-1][i+1]))   

# def all_flucts(k): # array of fluctations for each residue for mode k, for plotting
#     flucts = []
    
#     for i in range(1,731,3): # for range(1,n,3); n = (number of residues)*3 - 1
#         flucts.append(fluct(k,i))  

#     return flucts

# # # plot several modes
# for i in range(1,4): # range(1,n); plot first n-1 modes (remember all_fluct() defines the first mode as k=1, not k=0
#     plt.scatter(np.linspace(31,275,244),all_flucts(i),marker='o',s = 80)
#     plt.plot(np.linspace(31,275,244),all_flucts(i),linewidth = 2)

# # # plot single mode of interest
# # # plt.scatter(np.linspace(31,275,244),all_flucts(3),marker='o',s=4)
# # # plt.plot(np.linspace(31,275,244),all_flucts(3))

# # ### plotting square fluctations per residue
# plt.legend(['PC 1','PC 2','PC 3'],shadow = False) ## change legend for number of modes
# plt.xlabel('Residue', fontsize = 24)
# plt.ylabel('Square Fluctuations (|\u0394$R_i|^2$)', fontsize = 22)
# plt.ylim(0,0.004)  # ylim changed to ignore masive fluctations at terminals 
# plt.title('oxa24-WT MD-PCA', fontsize = 24)    # edit title
# plt.xticks(np.linspace(30,270,25),fontsize = 10)
# plt.show()

# # ### calculate degree of collectivity per mode according to Bahar et al. 2010
# def collectivity(k):
#     alpha = 1/(np.sum(all_flucts(k)))
#     return (3/len(eigvecs))*np.exp(-np.sum(np.multiply(all_flucts(k),alpha)*np.log(np.multiply(all_flucts(k),alpha))))

# degrs_coll = []
# for i in range(1,21):  ## set range according to number of modes
#     degrs_coll.append(collectivity(i))

# ## plotting degress of collectivity per mode
# degrs_coll = [ele/sum(degrs_coll) for ele in degrs_coll[:10]]
# degrs_coll = [ele/max(degrs_coll) for ele in degrs_coll]

# fig, ax = plt.subplots()

# ax2 = ax.twinx()
# ax.bar(np.linspace(1,10,10),degrs_coll,color = 'green')
# epvar = [ele/sum(eigvals) for ele in eigvals[:10]]
# ax2.scatter(np.linspace(1,10,10),epvar,marker = 's', color = 'black')
# ax2.plot(np.linspace(1,10,10),epvar,linewidth = 3, color = 'black')

# ax.set_xlabel('Mode',fontsize = 22)
# ax.set_ylabel('Relative Degree of Collectivity', fontsize = 22)
# ax.set_xticks(np.linspace(1,10,10))
# ax.set_title('MD-PCA oxa24-R261S', fontsize = 24)
# ax2.set_ylabel('Explained Variance', fontsize = 22)
# # ax.legend(['degree of collectivity'])
# # ax2.legend(['explained variance'],bbox_to_anchor=(0.47, 0., 0.5, 0.93))
# # plt.show()

# plt.bar([1,2,3,4,5],[1,2,3,4,5],color = 'green')
# plt.scatter([1,2,3],[1,2,3],marker = 's', color = 'black')
# plt.legend(['degree of collectivity','explained variance'])
# # plt.show()

if str(sys.argv[1] == 'change'):
    traj1 = pt.load('oxa24_fullprodby10.nc','oxa24-WT.parm7') ### trajectory (.nc), parameters (.parm7)
    data1 = pt.pca(traj1, mask='@CA', n_vecs = 10)
    traj2 = pt.load('oxa24_r261s_fullprodby10.nc','oxa24-R261S.parm7') ### trajectory (.nc), parameters (.parm7)
    data2 = pt.pca(traj2, mask='@CA', n_vecs = 10)

    eigvals1 = data1[1][0] # WT is 1, R261S is 2
    eigvecs1 = data1[1][1]
    eigvals2 = data2[1][0]
    eigvecs2 = data2[1][1]
    
    def fluct1(k,i): 
        return (1/eigvals1[k-1])*(np.dot(eigvecs1[k-1][i-1],eigvecs1[k-1][i-1]) + np.dot(eigvecs1[k-1][i],eigvecs1[k-1][i]) + np.dot(eigvecs1[k-1][i+1],eigvecs1[k-1][i+1]))   

    def all_flucts1(k): # array of fluctations for each residue for mode k, for plotting
        flucts1 = []
        
        for i in range(1,731,3): # for range(1,n,3); n = (number of residues)*3 - 1
            flucts1.append(fluct1(k,i))  

        return flucts1

    def fluct2(k,i): 
        return (1/eigvals2[k-1])*(np.dot(eigvecs2[k-1][i-1],eigvecs2[k-1][i-1]) + np.dot(eigvecs2[k-1][i],eigvecs2[k-1][i]) + np.dot(eigvecs2[k-1][i+1],eigvecs2[k-1][i+1]))   

    def all_flucts2(k): # array of fluctations for each residue for mode k, for plotting
        flucts2 = []
        
        for i in range(1,731,3): # for range(1,n,3); n = (number of residues)*3 - 1
            flucts2.append(fluct2(k,i))  

        return flucts2

    flucts11list = all_flucts1(1) # WT is 1, R261S is 2, so flucts22 would be R261S mode 2
    flucts21list = all_flucts2(1)
    flucts12list = all_flucts1(2)
    flucts22list = all_flucts2(2)
    flucts13list = all_flucts1(3)
    flucts23list = all_flucts2(3)
    flucts14list = all_flucts1(4)
    flucts24list = all_flucts2(4)

    change_flucts = []
    for i in range(0,len(flucts11list)):
        change_flucts.append(flucts24list[i] - flucts13list[i]) ## change name of lists here for comparison of different modes
        
    # plot single mode of interest
    plt.scatter(np.linspace(31,275,244),change_flucts,marker='o',s=80)
    plt.plot(np.linspace(31,275,244),change_flucts,linewidth = 3,color = 'black')

    plt.xlabel('Residue', fontsize = 24)
    plt.ylabel('Change in Square Fluctuations', fontsize = 24)
    plt.ylim(2*min(change_flucts),2*max(change_flucts))  # ylim changed to ignore masive differences at terminals 
    # plt.title('oxa24 (R261S  - WT) change in square fluctuations', fontsize = 20)    # edit title
    plt.xticks(np.linspace(30,270,25),fontsize = 10)
    plt.show()

    cumsum = [ele/sum(eigvals1) for ele in eigvals1] # I'm getting alright at list comprehesions =)
    cumsum2 = [ele/sum(eigvals2) for ele in eigvals2]

    ## cumulative sum of explained variance
    # cumsum = np.cumsum(cumsum)
    # cumsum = np.insert(cumsum,0,0)
    # cumsum2 = np.cumsum(cumsum2)
    # cumsum2 = np.insert(cumsum2,0,0)
    # plt.scatter(np.linspace(0,10,11),cumsum,s = 120, marker='^')
    # plt.scatter(np.linspace(0,10,11),cumsum2,s = 120, marker='o')
    # plt.legend(['WT','R261S'],shadow=False, loc = 4)
    # plt.plot(cumsum,linewidth = 2, color = 'black')
    # plt.plot(cumsum2,linewidth = 2,color = 'black')
    # plt.xticks(np.linspace(1,10,10))
    # plt.title('Cumulative Sum of MD-PCA Explained Variences',fontsize=22)
    # plt.ylim(0,1.1)
    # plt.xlabel('eigenvector',fontsize=20)
    # plt.ylabel('explained variance',fontsize=20)
    # plt.show()

print('remember to reload amber if needed')
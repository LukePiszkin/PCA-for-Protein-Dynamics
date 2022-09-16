import sys
import numpy as np 
import matplotlib.pyplot as plt
from numpy import linalg  
from prody import *
from prody.proteins import header
from pylab import *
import matplotlib.font_manager as fm

# general plotting parameters
font_names = [f.name for f in fm.fontManager.ttflist]
mpl.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2
params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)

### build Anisotropic Network Model using the ProDy python module by Bahar et al. @ U Pitts 
pdb = parsePDB('AMBER/oxa24.pdb',subset='calpha') ## pdb file
anm = ANM('pdb ANM analysis')
calphas = pdb.select('protein and name CA')
anm.buildHessian(pdb)
anm.calcModes()

### save useful matrices
hessian = anm.getHessian()
kirchoff = anm.getKirchhoff()
eigvals = anm.getEigvals()
covar = anm.getCovariance()

### get the eigenvectors that correspond to the lowest 20 modes
eigvecs = []
for i in range(0,20):
    eigvecs.append(anm[i].getEigvec())

### calculate square fluctuations of residue i by mode k, according to Bahar et al. 2010
### summing the norm of each cartesian component for a particular residue
def fluct(k,i): 
    return (1/eigvals[k-1])*(np.dot(eigvecs[k-1][i-1],eigvecs[k-1][i-1]) + np.dot(eigvecs[k-1][i],eigvecs[k-1][i]) + np.dot(eigvecs[k-1][i+1],eigvecs[k-1][i+1]))   

def all_flucts(k): # array of fluctations for each residue for mode k, for plotting
    flucts = []
    
    for i in range(1,719,3): # for range(1,n,3); n = (number of residues)*3 - 1
        flucts.append(fluct(k,i))  

    return flucts

# plot multiple modes
for i in range(1,4): # range(1,n); plot first n-1 modes
    plt.scatter(np.linspace(35,275,240),all_flucts(i),marker='o',s=4)
    plt.plot(np.linspace(35,275,240),all_flucts(i))

# # plot single mode of interest
# plt.scatter(np.linspace(35,275,240),all_flucts(3),marker='o',s=4)
# plt.plot(np.linspace(35,275,240),all_flucts(3))

plt.legend(['mode 1','mode 2','mode 3'], shadow = True) ## change legend for number of modes
plt.xlabel('Residue',fontsize = 24)
plt.ylabel('Square Fluctuations (|\u0394$R_i|^2$)',fontsize = 22)
plt.ylim(0,0.2)  # ylim changed to ignore masive fluctations at terminals 
plt.title('oxa24-WT Truncated',fontsize = 24)    # edit title
plt.xticks(np.linspace(30,275,50),fontsize = 8)
plt.show()


### calculate degree of collectivity per mode according to Bahar et al. 2010
def collectivity(k):
    alpha = 1/(np.sum(all_flucts(k)))
    return (3/len(eigvecs))*np.exp(-np.sum(np.multiply(all_flucts(k),alpha)*np.log(np.multiply(all_flucts(k),alpha))))

# degrs_coll = []
# for i in range(1,21):
#     degrs_coll.append(collectivity(i))

# fig, ax = plt.subplots()
# pps = ax.bar(np.linspace(1,20,20),degrs_coll,color = 'green')

# b = 0
# for p in pps:
#     height = p.get_height()
#     ax.annotate('{}'.format(eigvals[b].round(2)),
#     xy=(p.get_x() + p.get_width() / 2, height),
#     xytext=(0, 3), # 3 points vertical offset
#     textcoords="offset points",
#     ha='center', va='bottom')
#     b = b + 1

# ax.set_xlabel('Mode',fontsize = 24)
# ax.set_ylabel('Degree of Collectivity', fontsize = 24)
# ax.set_xticks(np.linspace(1,20,20))
# ax.set_title('oxa24-WT Truncated', fontsize = 24)
# plt.show()

plt.scatter(np.linspace(1,20,20),eigvals,color = 'black', s = 100, marker = 'o',alpha = 0.5)
plt.xlabel('mode', fontsize = 24)
plt.ylabel('eigenvalue', fontsize = 24)
plt.xticks(np.linspace(1,20,20))
plt.title('Scree plot of NMA for oxa24-WT', fontsize = 24)
plt.show()

### write modes into trajectory files for visualizing motion in VMD with NMWiz
# anm[:n] writes the first n
# writeNMD('oxa24-WT_truncated.nmd', anm[:2], calphas)
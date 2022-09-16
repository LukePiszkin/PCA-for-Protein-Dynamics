from pickle import STACK_GLOBAL
from re import M
import numpy as np 
from sklearn.decomposition import PCA
from sklearn import preprocessing
import xlrd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
from sklearn.model_selection import train_test_split

# style stuff for plotting
font_names = [f.name for f in fm.fontManager.ttflist]
mpl.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.linewidth'] = 2
params = {'mathtext.default': 'regular' }          
plt.rcParams.update(params)

CSPs = []
residues = []
times = []

loc = ('oxa24-WT_CSP_all3.xls')
wb = xlrd.open_workbook(loc)
sheet = wb.sheet_by_index(0)

CSPs2 = []
residues2 = []
times2 = []

loc2 = ('oxa24-R261S_CSP_JWP.xls')
wb2 = xlrd.open_workbook(loc2)
sheet2 = wb2.sheet_by_index(0)

## WT
for i in range(0,sheet.nrows):
    if i != 0:
        CSPs.append(sheet.row_values(i))
    for j in range(0,sheet.ncols):
        if i == 0 and j != 0:
            residues.append(sheet.cell_value(i,j))

        if j == 0 and i != 0:
            times.append(sheet.cell_value(i,j))

for i in range(0,len(CSPs)):
    CSPs[i].pop(0)

## Mutant
for i in range(0,sheet2.nrows):
    if i != 0:
        CSPs2.append(sheet2.row_values(i))
    for j in range(0,sheet2.ncols):
        if i == 0 and j != 0:
            residues2.append(sheet2.cell_value(i,j))

        if j == 0 and i != 0:
            times2.append(sheet2.cell_value(i,j))

for i in range(0,len(CSPs2)):
    CSPs2[i].pop(0)

## PCA of WT
scaler = preprocessing.StandardScaler().fit(CSPs)
CSPscaled = scaler.transform(CSPs)
pca = PCA()
projected = pca.fit_transform(CSPscaled)
pca.fit_transform(CSPscaled)
eigvecs = pca.components_
eigvals = pca.explained_variance_
# print(pca.score(CSPscaled))

## PCA of Mutant
scaler2 = preprocessing.StandardScaler().fit(CSPs2)
CSPscaled2 = scaler2.transform(CSPs2)
pca2 = PCA()
projected2 = pca.fit_transform(CSPscaled2)
pca2.fit_transform(CSPscaled2)
eigvecs2 = pca2.components_
eigvals2 = pca2.explained_variance_
# print(pca2.score(CSPscaled2))

## scree plot
plt.plot(np.linspace(1,20,20),[ele/np.sum(eigvals) for ele in  eigvals[:20]],marker = 'o',color = 'blue')
plt.plot(np.linspace(1,20,20),[ele/np.sum(eigvals2) for ele in  eigvals2[:20]],marker = 'o', color = 'orange')
plt.title('Scree Plot of oxa24 CSPs',fontsize = 22)
plt.xlabel('eigenvector',fontsize = 24)
plt.ylabel('eigenvalue',fontsize=24)
plt.xticks(np.linspace(1,20,20))
plt.legend(['WT','R261S'],shadow=True)
plt.title('Eigenvalue vs Eigenvector, centered and scaled, normalized y axis')
plt.show()

## cumulative sum plot
eigvalsum = pca.explained_variance_ratio_
eigvalsum2 = pca2.explained_variance_ratio_
cumsum = np.cumsum([ele/np.sum(eigvals) for ele in  eigvals[:20]])
cumsum = np.insert(cumsum,0,0)
cumsum2 = np.cumsum([ele/np.sum(eigvals2) for ele in  eigvals2[:20]])
cumsum2 = np.insert(cumsum2,0,0)
plt.clf()
plt.scatter(np.linspace(0,20,21),cumsum, s = 120, marker='^')
plt.scatter(np.linspace(0,20,21),cumsum2,s = 120, marker='o')
plt.legend(['WT','R261S'],shadow=False, loc = 4)
plt.plot(cumsum,linewidth = 2, color = 'black')
plt.plot(cumsum2,linewidth = 2,color = 'black')
plt.xticks(np.linspace(1,20,20))
plt.title('Cumulative Sum of oxa24 CSPs',fontsize=22)
plt.ylim(0,1.1)
plt.xlabel('eigenvector',fontsize=20)
plt.ylabel('explained variance',fontsize=20)
# plt.show()

## projections plot
# plt.scatter(projected[:,0],projected[:,1])
# plt.xlabel('component 1')
# plt.ylabel('component 2')
# plt.show()

## Square Fluctionations for residue induce by the kth principal component
def fluct(k,i): 
    return (1/eigvals[k-1])*np.dot(eigvecs[k-1][i-1],eigvecs[k-1][i-1])

def all_flucts(k): # array of fluctations for each residue for mode k, for plotting
    flucts = []
    
    for i in range(1,len(residues)+1): # for range(1,n,3); n = (number of residues)*3 - 1
        flucts.append(fluct(k,i))  

    return flucts

def fluct2(k,i): 
    return (1/eigvals2[k-1])*np.dot(eigvecs2[k-1][i-1],eigvecs2[k-1][i-1])

def all_flucts2(k): # array of fluctations for each residue for mode k, for plotting
    flucts2 = []
    
    for i in range(1,len(residues2)+1): # for range(1,n,3); n = (number of residues)*3 - 1
        flucts2.append(fluct2(k,i))  

    return flucts2

plt.clf()

## plot several modes
# for i in range(1,2): # range(1,n); plot first n-1 modes (remember all_fluct() defines the first mode as k=1, not k=0
#     plt.scatter(np.linspace(1,len(residues),len(residues)),all_flucts2(i),marker='o',s=80)
    
# plt.legend(['PC 1'], shadow=False) ## change legend for number of modes

# for i in range(1,2):
#     plt.plot(np.linspace(1,len(residues),len(residues)),all_flucts2(i),color = 'black')

mode = 1 # change mode to plot
full_residues = []
n = 0
# for i in range(31,275):
#     if n >= 28:
#         full_residues.append(0)
#     elif len(residues[n]) == 6:
#         if int(residues[n][1:3]) == i:
#             full_residues.append(all_flucts(mode)[n]) # change number in all_flucts for different mode
#             n = n + 1
#         else:
#             full_residues.append(0)
            
#     elif len(residues[n]) == 7:
#         if int(residues[n][1:4]) == i:
#             full_residues.append(all_flucts(mode)[n])  # change number in all_flucts for different mode
#             n = n + 1
#         else:
#             full_residues.append(0)

# full_residues2 = []
# n = 0
# for i in range(31,275):
#     if n >= 28:
#         full_residues2.append(0)
#     elif len(residues[n]) == 6:
#         if int(residues[n][1:3]) == i:
#             full_residues2.append(all_flucts2(mode)[n]) # change number in all_flucts for different mode
#             n = n + 1
#         else:
#             full_residues2.append(0)
            
#     elif len(residues[n]) == 7:
#         if int(residues[n][1:4]) == i:
#             full_residues2.append(all_flucts2(mode)[n])  # change number in all_flucts for different mode
#             n = n + 1
#         else:
#             full_residues2.append(0)
            
## plot single modes of interest
# plt.scatter(np.linspace(1,len(residues),len(residues)),marker='o',s=4)
# plt.plot(np.linspace(1,len(residues),len(residues)),all_flucts(1))
# plt.scatter(np.linspace(1,len(residues2),len(residues2)),all_flucts2(1),marker='o',s=4)
# plt.plot(np.linspace(1,len(residues2),len(residues2)),all_flucts2(1))

# plt.scatter(np.linspace(31,275,244),full_residues,marker='o',s=120)
# plt.scatter(np.linspace(31,275,244),full_residues2,marker='o',s=120)

# plt.legend(['WT','R261S'])

# plt.plot(np.linspace(31,275,244),full_residues)
# plt.plot(np.linspace(31,275,244),full_residues2)

## plotting square fluctations per residue
plt.xlabel('Residue', fontsize = 24)
plt.ylabel('Square Fluctuations (|\u0394$q_i|^2$)', fontsize = 24)
# plt.ylim(0,0.08) 
# plt.xticks(np.linspace(30,275,50),fontsize = 8)
plt.title('oxa24 R261S autoscaled CSP-PCA', fontsize = 22) # edit title

xtix1 = [ele[:3] for ele in residues if len(ele) == 3]
xtix2 = [ele[:4] for ele in residues if len(ele) == 4]
xtix = xtix1 + xtix2 

plt.xticks(np.linspace(1,len(residues),len(residues)),xtix,fontsize = 8)
# plt.show()
plt.clf()

def collectivity(k):
    alpha = 1/(np.sum(all_flucts(k)))
    return (3/len(eigvecs))*np.exp(-np.sum(np.multiply(all_flucts(k),alpha)*np.log(np.multiply(all_flucts(k),alpha))))

def collectivity2(k):
    alpha2 = 1/(np.sum(all_flucts2(k)))
    return (3/len(eigvecs2))*np.exp(-np.sum(np.multiply(all_flucts2(k),alpha2)*np.log(np.multiply(all_flucts2(k),alpha2))))

degrs_coll = []
for i in range(1,21):  ## set range according to number of modes
    degrs_coll.append(collectivity(i)) ## change name here to switch from WT to R261S (collectivity2)

## plotting degress of collectivity per mode
degrs_coll = [ele/max(degrs_coll) for ele in degrs_coll] ## normalize max k to 1
plt.bar(np.linspace(1,20,20),degrs_coll, color = 'green')
plt.xlabel('Mode',fontsize = 22)
plt.ylabel('relative k', fontsize = 22)
plt.title('Relative Degree of Collectivity for oxa24 R261S', fontsize = 22)
plt.xticks(np.linspace(1,20,20))
plt.show()

## plot collectivity and eigenvalues on same plot
# fig, ax = plt.subplots()

# ax2 = ax.twinx()
# ax.bar(np.linspace(1,20,20),degrs_coll,color = 'green')
# epvar = [ele/sum(eigvals) for ele in eigvals[:20]] ## change name here between mutants
# ax2.scatter(np.linspace(1,20,20),epvar,marker = 's', color = 'black')
# ax2.plot(np.linspace(1,20,20),epvar,linewidth = 3, color = 'black')

# ax.set_xlabel('Mode',fontsize = 22)
# ax.set_ylabel('Relative Degree of Collectivity', fontsize = 22)
# ax.set_xticks(np.linspace(1,10,10))
# ax.set_title('MD-PCA oxa24-WT', fontsize = 24)
# ax2.set_ylabel('Explained Variance', fontsize = 22)
# plt.show()

## for Chimera Render by Attribute
# for i in range(0,len(residues)):
#     if len(residues[i]) == 6:
#         print('\t:' + str(int(residues[i][1:3]) - 31) + '\t' + str('{:.10f}'.format(all_flucts2(2)[i]))) # change mode by argument of all_flucts
#     if len(residues[i]) == 7:
#         print('\t:' + str(int(residues[i][1:4]) - 31) + '\t' + str('{:.10f}'.format(all_flucts2(2)[i])))
import numpy as np
from estruturas import NARMAXPol

buck_id = np.loadtxt('buckdec_id.dat')
buck_val = np.loadtxt('buckdec_val.dat')         
uid = buck_id[:,1]
yid = buck_id[:,2]
uval = buck_val[:,1]
yval = buck_val[:,2]
ny=3 # Atraso máximo da saída y
nu=np.array([3]) # Atrasos máximos entradas: n_u1,n_u2,...,n_um (m - número de entradas)
n_lin=3 # Não linearidade máxima
narmax = NARMAXPol(uid, yid, nu, ny, n_lin)

#taxa de redução de erro
print(narmax.ERR())
#matriz de termos candidatos
print(narmax.candidatos)
#matriz de regressores candidatos
print(narmax.psi_candidatos)
#matriz de regressores candidatos ordenados de forma decrescente pelo ERR
print(narmax.psi_err)
#critério de informação seguindo a ordem de inclusão pelo ERR
print(narmax.InfoCriteria('aic'))
print(narmax.InfoCriteria('bic'))
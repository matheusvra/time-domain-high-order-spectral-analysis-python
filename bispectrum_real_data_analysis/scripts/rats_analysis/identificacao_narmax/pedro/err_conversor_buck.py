# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 19:01:00 2021

@author: pedro
"""
import numpy as np


def termos(ny,nu,n_lin,cte):
    
    # Entradas:
    
    #ny: atraso máximo da saída
    #nu: atraso(s) máximo(s) entradas(s) - exemplo nu=np.array([2,3,4]); nu1=2, nu2=3, nu3=4
    #n_lin: não linearidade máxima (1 para o caso linear)    
    #cte: inserir regressor constante? (0-Não, 1-Sim)
    
    # Saída:
    
    #termos: regressores na forma [y(k-1) ... y(k-ny) u1(k-1)...u1(k-nu1) ...]

    numRegLin = ny+np.sum(nu) # Número de regressores lineares (sem contar com o regressor constante)
        
    termos=np.zeros((1,numRegLin)) # Regressor constante
    
    for i in range(n_lin):   
        x_1=partitions_dp(i+1)
        
        for j in range(len(x_1)):
            
            if numRegLin>=np.array(x_1[j]).shape[0]:
            
                x_2=np.array(x_1[j])
                
                if len(x_2)<numRegLin: # Completando x_2 com zeros
                    x_2=np.concatenate((x_2,np.zeros(numRegLin-len(x_2))),axis=0)
                    
                x_3=np.array(list(unique_permutations(x_2)))
                x_4=np.flip(x_3,axis=0)
                
                termos=np.concatenate((termos,x_4),axis=0)       
           
    
    # Retirando regressor constante caso cte != 1
    if cte != 1:
        termos=np.delete(termos,0,0)
        
    return termos

def regressores(yid,uid,ny,nu,termos):
    
    # Entradas:
    
    #yid: saída
    #uid: entrada(s)
    #ny: atraso máximo da entrada
    #nu: atraso(s) máximo(s) entradas(s) - exemplo nu=np.array([2,3,4]); nu1=2, nu2=3, nu3=4
    #termos: regressores na forma [y(k-1) ... y(k-ny) u1(k-1)...u1(k-nu1) ...]
    
    # Saída:
    
    #psi: matriz psi com todos os regressores incados em termos
    
    # Observações:
    
    # Somento para o caso SISO ou MISO
    
    # Determinando atraso máximo
    nmax=np.amax(np.array([ny,np.amax(nu)]))

    # Número de amostras
    N=yid.shape[0]-nmax

    # regy: todos os regressores lineares de yid
    regy=np.zeros((N,ny))
    for i in range(ny):
        regy[:,i]=yid[nmax-i-1:-i-1]
    
    # regu: todos os regressores lineares de uid
    regu=np.zeros((N,np.sum(nu)))
    k=0
    # Caso SISO
    if nu.shape[0] == 1:
        for i in range(nu[0]):
            regu[:,k]=uid[nmax-i-1:-i-1]
            k=k+1
    # Caso MISO
    else: # nu.shape[0]>1 (mais do que uma entrada)
        for i in range(nu.shape[0]):
            for j in range(nu[i]):
                regu[:,k]=uid[nmax-j-1:-j-1,i]
                k=k+1
    
    reg=np.concatenate((regy,regu),axis=1) # reg: todos os regressores lineares
    
    numRegLin = ny+np.sum(nu) # número de regressors lineares (sem contar com o regressor constante)
    numReg = termos.shape[0] # número de regressores candidatos para o espaço de regressores   
    
    # Matriz psi com todos os regressores candidatos
    psi=np.ones((N,numReg))
    for i in range(numReg):
        for j in range(numRegLin):
            if termos[i,j] == 1:
                psi[:,i]=np.multiply(psi[:,i],reg[:,j])
            elif termos[i,j] > 1:
                psi[:,i]=np.multiply(psi[:,i],np.power(reg[:,j],termos[i,j]))
                
    return psi



def err(yid,ny,nu,psi):
    
    # Entradas:
    
    #yid: entrada
    #ny: atraso máximo da entrada
    #nu: atraso(s) máximo(s) entradas(s) - exemplo nu=np.array([2,3,4]); nu1=2, nu2=3, nu3=4
    #psi: matriz psi com os regressores candidatos

    # Saídas
    
    #err[:,0] ordem dos regressores com maior ERR
    #err[:,1] valor da ERR na respectiva iteração de acordo com o regressor

    # Determinando atraso máximo
    nmax=np.amax(np.array([ny,np.amax(nu)]))
    
    # Saída para cálculo da ERR
    y=yid[nmax:]
        
    N=psi.shape[0] # Número de amostras
    numReg=psi.shape[1] # Número de regressores
     
    # Matriz psi para cálcular a ERR (sem alterar a psi de entrada)
    
    psi_err=np.copy(psi)
    
    # Variáveis que podem ser retornadas pela função (err_ite e ordem_reg)
    
    err_ite=np.zeros((numReg,numReg)) # err a cada iteração  [ERR | iteração]
    ordem_ite=np.zeros((numReg,numReg+1)) # ordem dos regresores com a ERR de cada iteração
    ordem_ite[:,0]=np.arange(numReg) # primeira iteração, ordem: 0,1,2,...,(numReg-1)
    
    indice=np.arange(numReg) # índice de controle dos regressores
    
    err=np.zeros((numReg,2)) # regressor com maior ERR de cada iteração na sequência [reg | ERR]
    
    
    for i in range (numReg):
               
        if i==0: # primeira iteração (não existe necessidade de ortogonalizar)
            for j in range(numReg):
                err_ite[j,i]=np.power(np.dot(psi_err[:,j],y),2)/(np.dot(psi_err[:,j],psi_err[:,j])*np.dot(y,y))
             
            err[i,0]=indice[np.argmax(err_ite[:,i])] # indice do regressor com maior ERR
            err[i,1]=np.amax(err_ite[:,i]) # valor do maior ERR        
            
            # Alterando matriz psi_err para próxima iteração
            
            aux_1=np.copy(psi_err[:,np.argmax(err_ite[:,i])]) # regressor de maior ERR
            aux_2=np.copy(psi_err[:,i]) # Regressor com a posição que vai ser substituida pelo de maior ERR
            
            # Trocando as colunas na matriz psi_err
            psi_err[:,i]=aux_1
            psi_err[:,np.argmax(err_ite[:,i])]=aux_2
    
            # Alterando os índices dos regressores trocados
            
            aux_0=np.copy(indice)
            indice[i]=aux_0[np.argmax(err_ite[:,i])] # indice da iteração i recebe indice de reg com maior ERR
            indice[np.argmax(err_ite[:,i])]=aux_0[i] # indice do reg com maior ERR troca do o indice da iteração i
            
            ordem_ite[:,i+1]=np.copy(indice) 
            
        else: # próximas iterações (nesse caso é necessário ortogonalizar)
                             
            w_psi=np.zeros((N,i+1)) # matriz de regressores para ortogonalizar
            
            # Inserido regressores já classificados na w_psi
            for j in range(i):           
                w_psi[:,j]=psi_err[:,j]
                
            # Cálculo da ERR
            for j in range(numReg-i):
                
                w_psi[:,i]=psi_err[:,j+i] # Adicionando o regressores que vai ser cáculado a ERR
                
                QR=np.linalg.qr(w_psi)   # Ortogonalizando a matriz de regressores
                
                w_ort=QR[0] # Matriz de regressores ortogonal
                
                # Cálculo ERR
                err_ite[j+i,i] = np.power(np.dot(w_ort[:,i],y),2)/np.dot(y,y) # obs: np.dot(w_ort[:,i],w_ort[:,i]) = 1 (omitido na conta)
            
            err[i,0]=indice[np.argmax(err_ite[:,i])] # indice do regressor com maior ERR
            err[i,1]=np.amax(err_ite[:,i]) # valor do maior ERR        
            
            # Alterando matriz psi_err para próxima iteração
            
            aux_1=np.copy(psi_err[:,np.argmax(err_ite[:,i])]) # regressor de maior ERR
            aux_2=np.copy(psi_err[:,i]) # Regressor com a posição que vai ser substituida pelo de maior ERR
            
            # Trocando as colunas na matriz psi_err
            psi_err[:,i]=aux_1
            psi_err[:,np.argmax(err_ite[:,i])]=aux_2
    
            # Alterando os índices dos regressores trocados
            
            aux_0=np.copy(indice)
            indice[i]=aux_0[np.argmax(err_ite[:,i])] # indice da iteração i recebe indice de reg com maior ERR
            indice[np.argmax(err_ite[:,i])]=aux_0[i] # indice do reg com maior ERR troca do o indice da iteração i
            
            ordem_ite[:,i+1]=np.copy(indice)
        
    ordem_ite=np.delete(ordem_ite,i+1,1) # numpy.delete(arr, obj, axis=None)   
    
    errdic = dict()
    errdic['ordem']=err[:,0].astype(int)
    errdic['valor']=err[:,1]
    

    return errdic

## Funções auxiliares
    
# Função utilizada em termos
# retorna todas as possibilidades de soma de determinado número n
def partitions_dp(n):
    partitions_of = []
    partitions_of.append([()])
    partitions_of.append([(1,)])
    for num in range(2, n+1):
        ptitions = set()
        for i in range(num):
            for partition in partitions_of[i]:
                ptitions.add(tuple(sorted((num - i, ) + partition)))
        partitions_of.append(list(ptitions))
    return partitions_of[n]

# Função utilizada em termos
# retorna todas as possibilidade de permutação de determinado array sem repetições
def unique_permutations(elements):
    if len(elements) == 1:
        yield (elements[0],)
    else:
        unique_elements = set(elements)
        for first_element in unique_elements:
            remaining_elements = list(elements)
            remaining_elements.remove(first_element)
            for sub_permutation in unique_permutations(remaining_elements):
                yield (first_element,) + sub_permutation
                

if __name__ == "__main__":
    #%%
    # Carregando dados (já decimados) 
                    
    buck_id = np.loadtxt('buckdec_id.dat')
    buck_val = np.loadtxt('buckdec_val.dat')
                    
    uid = buck_id[:,1]
    yid = buck_id[:,2]

    uval = buck_val[:,1]
    yval = buck_val[:,2]

    ny=2 # Atraso máximo da saída y
    nu=np.array([2]) # Atrasos máximos entradas: n_u1,n_u2,...,n_um (m - número de entradas)

    n_lin=2 # Não linearidade máxima

    termos=termos(ny,nu,n_lin,1)
    psi=regressores(yid,uid,ny,nu,termos)
    ferr=err(yid,ny,nu,psi)


    print(ferr['ordem'])
    print(ferr['valor'])
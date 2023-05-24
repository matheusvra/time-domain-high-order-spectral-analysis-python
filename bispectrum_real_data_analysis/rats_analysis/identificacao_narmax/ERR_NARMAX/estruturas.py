import numpy as np
import statsmodels.api as sm

class NARMAXPol():
    def __init__(self, u, y, nu, ny, l, cte=False):
        """
        NARMAX polinomial - MISO/ SISO
        Parâmetros:
            u: entrada
            y: saída
            nu: máximo atraso em u
            ny: máximo atraso em y
            l: grau de não linearidade
            cte: flag para inclusão de termo constante
        """
        self._u = u
        self._y = y
        self._nu = nu
        self._ny = ny
        self._l = l
        self._cte = cte

    def UniquePermutations(self, elements):
        """
        Função utilizada em GeraCandidatos
        retorna todas as possibilidade de permutação de determinado array sem repetições
        """
        if len(elements) == 1:
            yield (elements[0],)
        else:
            unique_elements = set(elements)
            for first_element in unique_elements:
                remaining_elements = list(elements)
                remaining_elements.remove(first_element)
                for sub_permutation in self.UniquePermutations(remaining_elements):
                    yield (first_element,) + sub_permutation

    def PartitionsDP(self, n):
        """
        Função utilizada em GeraCandidatos
        retorna todas as possibilidades de soma de determinado número n
        """
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
    
    def GeraCandidatos(self):
        """
        Gera uma matriz que indica os termos candidatos com todas as
        possíveis combinações de grau l das variáveis y(k-i), i=1,2,
        ...ny, e u(k-j), j=1,2,...,nu."""

        # Número de regressores lineares (sem contar com o regressor constante)
        numRegLin = self._ny+np.sum(self._nu) 
        # Regressor constante
        self.candidatos=np.zeros((1,numRegLin))

        for i in range(self._l):   
            x_1=self.PartitionsDP(i+1)
            for j in range(len(x_1)):
                if numRegLin>=np.array(x_1[j]).shape[0]:
                    x_2=np.array(x_1[j])
                    if len(x_2)<numRegLin: # Completando x_2 com zeros
                        x_2=np.concatenate((x_2,np.zeros(numRegLin-len(x_2))),axis=0)
                    x_3=np.array(list(self.UniquePermutations(x_2)))
                    x_4=np.flip(x_3,axis=0)
                    self.candidatos=np.concatenate((self.candidatos,x_4),axis=0)       
        # Retirando regressor constante caso cte == False
        if self._cte == False:
            self.candidatos=np.delete(self.candidatos,0,0)

    def GeraRegressoresCandidatos(self):
        """Constrói a matriz de regressores completa com todos os termos candidatos"""
        
        #Gera conjunto de regressores candidatos
        self.GeraCandidatos()
        # Determinando atraso máximo
        nmax=np.amax(np.array([self._ny,np.amax(self._nu)]))
        # Número de amostras
        N=self._y.shape[0]-nmax
        # regy: todos os regressores lineares de yid
        regy=np.zeros((N,self._ny))
        for i in range(self._ny):
            regy[:,i]=self._y[nmax-i-1:-i-1]
        # regu: todos os regressores lineares de uid
        regu=np.zeros((N,np.sum(self._nu)))
        k=0
        # Caso SISO
        if self._nu.shape[0] == 1:
            for i in range(self._nu[0]):
                regu[:,k]=self._u[nmax-i-1:-i-1]
                k=k+1
        # Caso MISO
        else: # nu.shape[0]>1 (mais do que uma entrada)
            for i in range(self._nu.shape[0]):
                for j in range(self._nu[i]):
                    regu[:,k]=self._u[nmax-j-1:-j-1,i]
                    k=k+1
        # reg: todos os regressores lineares
        reg=np.concatenate((regy,regu),axis=1) 
        # número de regressors lineares (sem contar com o regressor constante)
        numRegLin = self._ny+np.sum(self._nu)
        # número de regressores candidatos para o espaço de regressores 
        numReg = self.candidatos.shape[0]
        # Matriz psi com todos os regressores candidatos
        self.psi_candidatos=np.ones((N,numReg))
        for i in range(numReg):
            for j in range(numRegLin):
                if self.candidatos[i,j] == 1:
                    self.psi_candidatos[:,i]=np.multiply(self.psi_candidatos[:,i],reg[:,j])
                elif self.candidatos[i,j] > 1:
                    self.psi_candidatos[:,i]=np.multiply(self.psi_candidatos[:,i],
                                                         np.power(reg[:,j],self.candidatos[i,j]))
    
    def ERR(self):
        """Retorna o ERR de cada regressor de forma decrescente e o seu respectivo índice"""

        # Determinando atraso máximo
        nmax=np.amax(np.array([self._ny,np.amax(self._nu)]))
        # Saída para cálculo da ERR
        y=self._y[nmax:]
        #Gera os regressores candidatos
        self.GeraRegressoresCandidatos()
        # Número de amostras
        N=self.psi_candidatos.shape[0]
        # Número de regressores
        numReg=self.psi_candidatos.shape[1]
        # Matriz psi ordenada pelo ERR (sem alterar a psi de candidatos)
        self.psi_err=np.copy(self.psi_candidatos)
        # Variáveis que podem ser retornadas pela função (err_ite e ordem_reg)
        err_ite=-1*np.ones((numReg,numReg)) # err a cada iteração  [ERR | iteração]
        ordem_ite=np.zeros((numReg,numReg+1)) # ordem dos regresores com a ERR de cada iteração
        ordem_ite[:,0]=np.arange(numReg) # primeira iteração, ordem: 0,1,2,...,(numReg-1)
        indice=np.arange(numReg) # índice de controle dos regressores
        err=np.zeros((numReg,2)) # regressor com maior ERR de cada iteração na sequência [reg | ERR]
        for i in range (numReg):
            if i==0: # primeira iteração (não existe necessidade de ortogonalizar)
                for j in range(numReg):
                    err_ite[j,i]=(np.power(np.dot(self.psi_err[:,j],y),2)/
                                          (np.dot(self.psi_err[:,j],self.psi_err[:,j])*np.dot(y,y)))
                err[i,0]=indice[np.argmax(err_ite[:,i])] # indice do regressor com maior ERR
                err[i,1]=np.amax(err_ite[:,i]) # valor do maior ERR        
                # Alterando matriz psi_err para próxima iteração
                # regressor de maior ERR
                aux_1=np.copy(self.psi_err[:,np.argmax(err_ite[:,i])])
                # Regressor com a posição que vai ser substituida pelo de maior ERR
                aux_2=np.copy(self.psi_err[:,i]) 
                # Trocando as colunas na matriz psi_err
                self.psi_err[:,i]=aux_1
                self.psi_err[:,np.argmax(err_ite[:,i])]=aux_2
                # Alterando os índices dos regressores trocados
                aux_0=np.copy(indice)
                # indice da iteração i recebe indice de reg com maior ERR
                indice[i]=aux_0[np.argmax(err_ite[:,i])]
                # indice do reg com maior ERR troca do o indice da iteração i
                indice[np.argmax(err_ite[:,i])]=aux_0[i] 
                
                ordem_ite[:,i+1]=np.copy(indice) 
                
            else: # próximas iterações (nesse caso é necessário ortogonalizar)       
                w_psi=np.zeros((N,i+1)) # matriz de regressores para ortogonalizar
                # Inserido regressores já classificados na w_psi
                for j in range(i):           
                    w_psi[:,j]=self.psi_err[:,j]
                # Cálculo da ERR
                for j in range(numReg-i):
                    # Adicionando o regressores que vai ser cáculado a ERR
                    w_psi[:,i]=self.psi_err[:,j+i] 
                    # Ortogonalizando a matriz de regressores
                    QR=np.linalg.qr(w_psi)   
                    # Matriz de regressores ortogonal
                    w_ort=QR[0] 
                    # Cálculo ERR
                    # obs: np.dot(w_ort[:,i],w_ort[:,i]) = 1 (omitido na conta)
                    err_ite[j+i,i] = np.power(np.dot(w_ort[:,i],y),2)/np.dot(y,y)
                err[i,0]=indice[np.argmax(err_ite[:,i])] # indice do regressor com maior ERR
                err[i,1]=np.amax(err_ite[:,i]) # valor do maior ERR        
                # Alterando matriz psi_err para próxima iteração
                # regressor de maior ERR
                aux_1=np.copy(self.psi_err[:,np.argmax(err_ite[:,i])])
                # Regressor com a posição que vai ser substituida pelo de maior ERR
                aux_2=np.copy(self.psi_err[:,i])
                # Trocando as colunas na matriz psi_err
                self.psi_err[:,i]=aux_1
                self.psi_err[:,np.argmax(err_ite[:,i])]=aux_2
                # Alterando os índices dos regressores trocados
                aux_0=np.copy(indice)
                # indice da iteração i recebe indice de reg com maior ERR
                indice[i]=aux_0[np.argmax(err_ite[:,i])]
                # indice do reg com maior ERR troca do o indice da iteração i 
                indice[np.argmax(err_ite[:,i])]=aux_0[i]
                
                ordem_ite[:,i+1]=np.copy(indice)
            
        ordem_ite=np.delete(ordem_ite,i+1,1) # numpy.delete(arr, obj, axis=None)   
        errdic = dict()
        errdic['ordem']=err[:,0].astype(int)
        errdic['valor']=err[:,1]
        return errdic

    def InfoCriteria(self, tipo='aic'):
        """Retorna do critério de informação para cada regressor
        da matriz psi_candidatos. A ordem de avaliação do critério de informação
        segue o ERR."""
        # Determinando atraso máximo
        nmax=np.amax(np.array([self._ny,np.amax(self._nu)]))
        # Saída para cálculo do AIC
        y=self._y[nmax:]
        # Número de regressores
        numReg=self.psi_err.shape[1]
        #AIC
        if tipo=='aic':
            AIC=[]
            for i in range(1,numReg+1):
                modelo=sm.OLS(y, self.psi_err[:,:i]).fit()
                AIC.append(modelo.aic)
            return np.array(AIC)
        elif tipo=='bic':
            #BIC
            BIC=[]
            for i in range(1,numReg+1):
                modelo=sm.OLS(y, self.psi_err[:,:i]).fit()
                BIC.append(modelo.bic)
            return np.array(BIC)
        return None



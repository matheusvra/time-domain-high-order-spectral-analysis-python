# -*- coding: utf-8 -*-
"""
Created on Tue Aug 10 19:01:00 2021
Modified on Sat Aug 21 2021

@author: pedro
@author: Matheus Ramos
"""
import numpy as np
import pandas as pd
from loguru import logger
import progressbar


class err:
    """Classe que contém o algoritmo ERR e os métodos que o algoritmo utiliza. Requer numpy as np.

    Raises:
        Exception: Caso os métodos sejam executados em ordem errada

    Runnging:
        Construct the class object and call the run method.
    """

    nmax: int
    nu: int
    ny: int
    n_lin: int
    y: np.ndarray
    u: np.ndarray
    cte: bool
    termos_matrix: np.ndarray
    psi: np.ndarray
    ferr: np.ndarray

    def __init__(
        self,
        ny: int,
        nu: int,
        n_lin: int,
        yid: np.ndarray,
        uid: np.ndarray,
        cte: bool = False,
        enable_progress_bar: bool = True,
    ) -> None:
        self.nu = nu if isinstance(nu, np.ndarray) else np.array([nu])
        self.ny = ny
        self.n_lin = n_lin
        self.y = yid
        self.u = uid
        self.cte = cte
        self.enable_progress_bar = enable_progress_bar

    def update(
        self,
        ny: int,
        nu: int,
        n_lin: int,
        yid: np.ndarray,
        uid: np.ndarray,
        cte: bool = False,
    ) -> None:
        self.nu = nu if isinstance(nu, np.ndarray) else np.array([nu])
        self.ny = ny
        self.n_lin = n_lin
        self.y = yid
        self.u = uid
        self.cte = cte

    def __termos(self) -> None:
        logger.info("Creating Terms")
        numRegLin = self.ny + np.sum(
            self.nu
        )  # Número de regressores lineares (sem contar com o regressor constante)

        self.termos_matrix = np.zeros((1, numRegLin))  # Regressor constante

        N = self.n_lin

        for i in range(N):
            x_1 = self.__partitions_dp(i + 1)

            for j in range(len(x_1)):
                if numRegLin >= np.array(x_1[j]).shape[0]:
                    x_2 = np.array(x_1[j])

                    if len(x_2) < numRegLin:  # Completando x_2 com zeros
                        x_2 = np.concatenate(
                            (x_2, np.zeros(numRegLin - len(x_2))), axis=0
                        )

                    x_3 = np.array(list(self.__unique_permutations(x_2)))
                    x_4 = np.flip(x_3, axis=0)

                    self.termos_matrix = np.concatenate(
                        (self.termos_matrix, x_4), axis=0
                    )

        # Retirando regressor constante caso cte != 1
        if not self.cte:
            self.termos_matrix = np.delete(self.termos_matrix, 0, 0)

    def __regressores(self) -> np.array:
        # Determinando atraso máximo
        self.nmax = np.amax(np.array([self.ny, np.amax(self.nu)]))

        # Número de amostras
        N = self.y.shape[0] - self.nmax

        # regy: todos os regressores lineares de yid
        regy = np.zeros((N, self.ny))
        for i in range(self.ny):
            regy[:, i] = self.y[self.nmax - i - 1 : -i - 1]

        # regu: todos os regressores lineares de uid
        regu = np.zeros((N, np.sum(self.nu)))
        k = 0
        # Caso SISO
        if self.nu.shape[0] == 1:
            for i in range(self.nu[0]):
                regu[:, k] = self.u[self.nmax - i - 1 : -i - 1]
                k = k + 1
        # Caso MISO
        else:  # nu.shape[0]>1 (mais do que uma entrada)
            for i in range(self.nu.shape[0]):
                for j in range(self.nu[i]):
                    regu[:, k] = self.u[i, self.nmax - j - 1 : -j - 1]
                    k = k + 1

        reg = np.concatenate((regy, regu), axis=1)  # reg: todos os regressores lineares

        numRegLin = self.ny + np.sum(
            self.nu
        )  # número de regressors lineares (sem contar com o regressor constante)
        numReg = self.termos_matrix.shape[
            0
        ]  # número de regressores candidatos para o espaço de regressores

        # Matriz psi com todos os regressores candidatos
        logger.info("Creating regressors - psi matrix")
        psi = np.ones((N, numReg))

        for i in range(numReg):
            for j in range(numRegLin):
                if self.termos_matrix[i, j] == 1:
                    psi[:, i] = np.multiply(psi[:, i], reg[:, j])
                elif self.termos_matrix[i, j] > 1:
                    psi[:, i] = np.multiply(
                        psi[:, i], np.power(reg[:, j], self.termos_matrix[i, j])
                    )

        self.psi = psi

    def __err(self):  # sourcery skip: raise-specific-error
        try:
            y = self.y[self.nmax :]
        except NameError as e:
            raise Exception(
                "Nome 'nmax' não definido, favor executar os métodos na ordem: termos -> regressores -> err"
            ) from e

        logger.info("Calculating ERR")

        N = self.psi.shape[0]  # Número de amostras
        numReg = self.psi.shape[1]  # Número de regressores

        # Matriz psi para cálcular a ERR (sem alterar a psi de entrada)

        psi_err = np.copy(self.psi)

        # Variáveis que podem ser retornadas pela função (err_ite e ordem_reg)

        err_ite = np.zeros((numReg, numReg))  # err a cada iteração  [ERR | iteração]
        ordem_ite = np.zeros(
            (numReg, numReg + 1)
        )  # ordem dos regresores com a ERR de cada iteração
        ordem_ite[:, 0] = np.arange(
            numReg
        )  # primeira iteração, ordem: 0,1,2,...,(numReg-1)

        indice = np.arange(numReg)  # índice de controle dos regressores

        err = np.zeros(
            (numReg, 2)
        )  # regressor com maior ERR de cada iteração na sequência [reg | ERR]

        if self.enable_progress_bar:
            bar = progressbar.ProgressBar(max_value=numReg)

        for i in range(numReg):
            if self.enable_progress_bar:
                bar.update(i + 1)

            if i == 0:  # primeira iteração (não existe necessidade de ortogonalizar)
                for j in range(numReg):
                    err_ite[j, i] = np.power(np.dot(psi_err[:, j], y), 2) / (
                        np.dot(psi_err[:, j], psi_err[:, j]) * np.dot(y, y)
                    )

            else:  # próximas iterações (nesse caso é necessário ortogonalizar)
                w_psi = np.zeros((N, i + 1))  # matriz de regressores para ortogonalizar

                # Inserido regressores já classificados na w_psi
                for j in range(i):
                    w_psi[:, j] = psi_err[:, j]

                # Cálculo da ERR
                for j in range(numReg - i):
                    w_psi[:, i] = psi_err[
                        :, j + i
                    ]  # Adicionando o regressores que vai ser cáculado a ERR

                    QR = np.linalg.qr(w_psi)  # Ortogonalizando a matriz de regressores

                    w_ort = QR[0]  # Matriz de regressores ortogonal

                    # Cálculo ERR
                    err_ite[j + i, i] = np.power(np.dot(w_ort[:, i], y), 2) / np.dot(
                        y, y
                    )  # obs: np.dot(w_ort[:,i],w_ort[:,i]) = 1 (omitido na conta)

            err[i, 0] = indice[
                np.argmax(err_ite[:, i])
            ]  # indice do regressor com maior ERR
            err[i, 1] = np.amax(err_ite[:, i])  # valor do maior ERR

            # Alterando matriz psi_err para próxima iteração

            aux_1 = np.copy(
                psi_err[:, np.argmax(err_ite[:, i])]
            )  # regressor de maior ERR
            aux_2 = np.copy(
                psi_err[:, i]
            )  # Regressor com a posição que vai ser substituida pelo de maior ERR

            # Trocando as colunas na matriz psi_err
            psi_err[:, i] = aux_1
            psi_err[:, np.argmax(err_ite[:, i])] = aux_2

            # Alterando os índices dos regressores trocados

            aux_0 = np.copy(indice)
            indice[i] = aux_0[
                np.argmax(err_ite[:, i])
            ]  # indice da iteração i recebe indice de reg com maior ERR
            indice[np.argmax(err_ite[:, i])] = aux_0[
                i
            ]  # indice do reg com maior ERR troca do o indice da iteração i

            ordem_ite[:, i + 1] = np.copy(indice)

        ordem_ite = np.delete(ordem_ite, i + 1, 1)  # numpy.delete(arr, obj, axis=None)

        errdic = {"ordem": err[:, 0].astype(int)}
        errdic["valor"] = err[:, 1]

        self.ferr = errdic

    ## Funções auxiliares

    # Função utilizada em termos
    # retorna todas as possibilidades de soma de determinado número n
    def __partitions_dp(self, n):
        partitions_of = [[()], [(1,)]]
        for num in range(2, n + 1):
            ptitions = set()
            for i in range(num):
                for partition in partitions_of[i]:
                    ptitions.add(tuple(sorted((num - i,) + partition)))
            partitions_of.append(list(ptitions))
        return partitions_of[n]

    # Função utilizada em termos
    # retorna todas as possibilidade de permutação de determinado array sem repetições
    def __unique_permutations(self, elements):
        if len(elements) == 1:
            yield (elements[0],)
        else:
            unique_elements = set(elements)
            for first_element in unique_elements:
                remaining_elements = list(elements)
                remaining_elements.remove(first_element)
                for sub_permutation in self.__unique_permutations(remaining_elements):
                    yield (first_element,) + sub_permutation

    def run(self, print_result=False):
        """Função que executa o método ERR

        Args:
            print_result (bool, optional): Flag para printar resultado. Defaults to False.

        Returns:
            (np.array): Matriz V

        """
        self.__termos()
        self.__regressores()
        self.__err()

        if print_result:
            output = pd.DataFrame.from_dict(self.ferr)
            print(output)

        termos = {}
        termosu_index = [0, 1]
        for i in range(np.sum(self.nu) + self.ny):
            # y comes first, then comes u
            if i < self.ny:
                termos[f"y(k-{i+1})"] = self.termos_matrix[:, i]
            elif self.nu.shape[0] == 1:
                termos[f"u(k-{i-self.nu[0]+1})"] = self.termos_matrix[:, i]
            else:
                termos[f"u({termosu_index[0]}, k-{termosu_index[1]})"] = self.termos_matrix[:, i]
                termosu_index[1] += 1
                if termosu_index[1] > self.nu[termosu_index[0]-1]:
                    termosu_index[0] += 1
                    termosu_index[1] = 1
                    
        termos = pd.DataFrame.from_dict(termos)

        logger.success("Done. ERR method finished.")

        return self.ferr, termos, self.psi

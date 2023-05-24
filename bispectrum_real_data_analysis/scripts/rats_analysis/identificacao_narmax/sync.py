import numpy as np
import matplotlib.pyplot as plt

def sync(residuo_modelo_livro, xi):
    # e(k)
    def error_sync(c, xi):
        e = np.zeros(len(xi))
        for k in range(1, len(xi)):
            e[k] = c*e[k-1] + xi[k-1]
        return e
    
    #J(c)
    def Jrms(N, k0, h):
        return np.sqrt((1/(N-k0+1))*(c*np.sum(np.power(h, 2)[k0:N])))

    # epsilon
    def epsilon(e):
    
        def norm(x):
            return (x-np.min(x))/np.linalg.norm(x-np.min(x))
    
        num = np.max(norm(e))
        den = num + 1
        return num/den
    
    # Execução
    c_vec = np.arange(0.01, 0.99, 0.1)

    e_vec_modelo_livro = np.zeros((len(c_vec),len(residuo_modelo_livro)))
    e_vec_modelo_estimado = np.zeros((len(c_vec),len(xi)))


    for k, c in enumerate(c_vec):
        e_vec_modelo_livro[k] = error_sync(c, residuo_modelo_livro)
        e_vec_modelo_estimado[k] = error_sync(c, xi)

    epsilon_vec_modelo_livro = np.zeros(len(c_vec))
    epsilon_vec_modelo_estimado = np.zeros(len(c_vec))

    for k in range(len(c_vec)):
        epsilon_vec_modelo_livro[k] = epsilon(e_vec_modelo_livro[k])
        epsilon_vec_modelo_estimado[k] = epsilon(e_vec_modelo_estimado[k])

    J_modelo_estimado = np.zeros(len(e_vec_modelo_estimado))
    J_modelo_livro = np.zeros(len(e_vec_modelo_livro))

    for k, c in enumerate(c_vec):
        J_modelo_estimado[k] = Jrms(N=len(xi), k0=0, h=c*e_vec_modelo_estimado[k])
        J_modelo_livro[k] = Jrms(N=len(residuo_modelo_livro), k0=0, h=c*e_vec_modelo_livro[k])

    plt.figure()
    plt.plot(c_vec, epsilon_vec_modelo_livro, label='modelo livro')
    plt.plot(c_vec, epsilon_vec_modelo_estimado, label='modelo estimado')
    plt.title('Erro de Sincronização Máximo Normalizado')
    plt.legend()
    plt.ylabel("$\epsilon_{m}$")
    caption = f"Figura 1: Erro de Sincronização Máximo Normalizado"
    xlabel = 'c'
    plt.xlabel(xlabel+"\n"+caption)

    plt.figure()
    plt.plot(c_vec, J_modelo_livro, label='modelo livro')
    plt.plot(c_vec, J_modelo_estimado, label='modelo estimado')
    plt.title('Custo de Sincronização')
    plt.legend()
    plt.ylabel("$J_{rms}$")
    caption = f"Figura 2: Custo de Sincronização"
    plt.xlabel(xlabel+"\n"+caption)

    plt.show()
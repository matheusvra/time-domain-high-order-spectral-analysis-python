# Classe que possui os métodos de correlação utilizados
import numpy as np

class Correlation:

    # função geradora do intervalo de confinça - default: 95%
    @classmethod
    def confidence(self, correlation, confidence='95'):
        if confidence == '95':
            stds = 1.96
        elif confidence == '99':
            stds = 3
        else:
            stds = 1
        return np.ones(len(correlation))*stds*np.std(correlation-np.mean(correlation))
    
    
    @classmethod
    def correlate(self, x1, x2, size_output='full', confidence='95', bilateral=True, norm=True):
        """Função que calcula a correlação cruzada entre Sinais 1 e 2

        Args:
            x1: Sinal 1
            x2: Sinal 2
            size_output: Tamanho da janela de dados de saída
            confidence (str, optional): Faixa de confiança. Defaults to '95'.
            bilateral (bool, optional): Retorna ambos os lados se true. Defaults to True.
            norm (bool, optional): Flag para normalizar caso true. Defaults to True.

        Returns:
            x_axis: eixo x (array)
            cross_correlation: Valores de correlação cruzada (array)
            confidence: intervalo de confiança (escalar)
        """

        size_output = len(x1) if size_output=='full' else size_output

        # Remove eixos de tamanho unitário
        x1 = np.squeeze(x1)
        x2 = np.squeeze(x2)
        
        # Calcula a correlação cruzada (normalizada ou não)
        if norm:
            cross_correlation = np.correlate((x1-np.mean(x1))/(np.std(x1)*len(x1)), \
                                     (x2-np.mean(x2))/(np.std(x2)), mode='full')
        else:
            cross_correlation = np.correlate(x1,x2, mode='full')
        confidence = self.confidence(cross_correlation, confidence)
        
        # Calcula o vetor de atrasos e fatia o vetor de correlação
        if bilateral:
            x_axis = np.arange(-int(size_output/2),int(size_output/2)+1,1)

            cross_correlation = cross_correlation[ \
            int(len(cross_correlation)/2)-int(size_output/2): \
            int(len(cross_correlation)/2)+int(size_output/2)+1]

            confidence = confidence[ \
            int(len(confidence)/2)-int(size_output/2): \
            int(len(confidence)/2)+int(size_output/2)+1]
        else:
            x_axis = np.arange(0,int(size_output/2)+1,1)

            cross_correlation = cross_correlation[int(size_output/2):size_output+1]

            confidence = confidence[int(size_output/2):size_output+1]
        
        # Retorna os vetores de amostras, correlação e intervalo de confiança
        return x_axis, cross_correlation, confidence
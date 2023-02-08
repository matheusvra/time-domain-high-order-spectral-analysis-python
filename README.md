
# Bispectrum Real Data Analysis

Este documento apresenta um tutorial sobre como instalar e configurar o ambiente pela primeira vez, instalar a atualizar as dependências, bem como rodar os scripts.

---

***Aviso 1***: Este tutorial foi criado e testado no Ubuntu. Se você não está usando uma distribuição baseada em Unix, alguns passos poderão ser diferentes e não estão cobertos neste documento.


## Importante

Este repositório é feito para aplicar os algoritmos do repositório [high_order_spectra_analysis](https://github.com/matheusvra/high_order_spectra_analysis) em dados reais coletados. Portanto, é necessário que os usuários desse repositório tenham familiaridade com o mesmo e possam debugar e identificar possíveis erros que ocorram no repositório que será utilizado, bem como os erros que ocorram nesse resultado, de modo que ambos estejam funcionando corretamente e estejam coerentes para que as análises sejam feitas adequadamente.

## Pré-requisitos

* Python 3.11
* Pip 3.11
* Poetry

# Configuração do Ambiente

## Instalando o python3.11+

Este rápido tutorial irá mostrar-lhe como instalar a versão mais recente
Python 3.11 no Ubuntu.

* Abra o terminal via Ctrl+Alt+T ou procurando por “Terminal” em lançador de aplicativos. Quando ele abrir, execute os comandos:

### Instalando o Python 3.11 no Ubuntu 20.04|18.04 usando o Apt Repo

```shell
sudo apt-get install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get install python3.11 -y
sudo apt-get install python3.11-distutils -y
```

Agora você tem pelo menos uma versão do Python 3.11 instalada, use o comando ***python*** para a versão 2.x (se ainda estiver instalada),
***python3*** para a versão principal usada no sistema operacional e ***python3.11*** para a versão 3.11.x. Talvez seja necessário tornar a versão ***3.11*** como a principal, o que pode ser feito seguindo os passos neste [tutorial](https://www.folkstalk.com/tech/set-python-3-as-default-ubuntu-with-code-examples/).

Para verificar se funcionou digite:

```shell
python3.11 --version
```

A saída no terminal deve ser algo do tipo:

Python 3.11.7

## Instalando pip

Pip é um sistema de gerenciamento de pacotes usado para instalar e gerenciar pacotes de software escritos em Python.

Recomenda-se instalar a versão mais recente do pip3.10:

```shell
sudo curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
```

e teste o resultado da instalação:

```shell
python3.11 -m pip --version
```

e após isso, dê um upgrade na versão do pip instalada

```shell
python3.11 -m pip install --upgrade pip
```

## Instalando o Poetry

Para instalar o Poetry, gerenciador de dependências utilizado neste repositório, basta executar o comando após a instalação do Python3.10 e pip:

```shell
sudo curl -sSL https://install.python-poetry.org | python3 -
```

## Versionamento de Dependências e Instalação

Todas as dependências utilizadas são versionadas no arquivo 'pyproject.toml', na raiz da pasta do projeto.

Os comandos são feitos utilizando o Poetry. Para instalar o projeto com a última versão do repositório, configure, crie e habilite o ambiente virtual:

Para configurar o caminho onde o ambiente virtual será criado, use o comando abaixo para que o ambiente seja criado na pasta do projeto e seja de fácil localização:

```shell
poetry config virtualenvs.in-project true
```

Após a configuração, dê o comando abaixo para criar o ambiente virtual:

```shell
poetry env use python3.11
```

E o comando abaixo para habilitar o ambiente virtual:

```shell
poetry shell
```

Após habilitado, é necessário instalar o projeto e suas dependências no ambiente vitual. Para isso, execute o comando:

```shell
poetry install
```

## Execução dos scripts

Para executar algum script, basta invocar o Python passando o caminho do script, ou usando o player intragrado da IDE utilizada, desde que o ambiente virtual esteja selecionado como interpretador Python. 

## Descrição abreviada dos scripts

### -- **download_data.ipynb**
Jupyter notebook para fazer o download dos arquivos. Isso foi necessário pois o github possui limite para armazenamento de arquivos grandes de forma gratuita. Rodar sempre após clonar o repositório.

### -- **load_data.m**
Método para ser executado no matlab, para conversão de dados para CSV visando compatibilidade.
Carrega os dados de um arquivo com extensão ".mat" no workspace do MATLAB.

### -- **run_and_process_data.m**
Método para ser executado no matlab, para conversão de dados para CSV visando compatibilidade.
Processa e converte os dados do workspace para um arquivo com extensão ".csv".

### -- **generate_bispectrum_of_data.py**
Lê um arquivo CSV com dados de algum experimento, e gera o bispectro (parametrizável) para cada um dos canais medidos, salvando os resultados em dois arquivos CSV (spectrum_df.csv e bispectrum_df.csv) na pasta bispectrum_real_data_analysis/data.

### -- **plot_data.py**
Plota os dados do arquivo bispectrum_df.csv em dois gráficos HTML iterativos, um para amplitude e um para fase.

### -- **check_tdbs.py**
Utiliza dados analíticos para validação do algoritmo *time domain bispectrum*, gerando plot interativo usando Plotly.

### -- **inferior_colliculus_and_amygdala_comparison.ipynb**
Notebook Jupyter com análise de coerência de fase a *phase clustering*, além de outros processamentos de sinal nos dados.

# Autores

* **Matheus Anjos** - [matheusvra@hotmail.com](mailto:matheusvra@hotmail.com)

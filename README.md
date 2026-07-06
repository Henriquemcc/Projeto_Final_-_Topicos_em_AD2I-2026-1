# Projeto Final - Tópicos em Análise de Dados, Descoberta de Conhecimento e Recuperação de Informação II: Introduction to Deep Learning - 2026-1

## Como executar os modelos

Para executar os modelos de aprendizado de máquina desenvolvidos, siga os seguintes passos:

### 1. Configure e ative um ambiente virtual python (opcional)

Em um Terminal (Prompt de Comando ou PowerShell), na pasta do projeto, digite o seguinte comando:

```shell
python -m venv .venv
```

Se o seu shell for o Terminal do Linux ou do Mac, na pasta do projeto, digite o seguinte comando:

```shell
source .venv/bin/activate
```

Se o seu shell for o Prompt de Comando do Windows, na pasta do projeto, digite o seguinte comando:

```shell
.venv\Scripts\activate.bat
```

Se o seu shell for o Terminal (ou PowerShell) do Windows, na pasta do projeto, digite o seguinte comando:

```shell
.\.venv\Scripts\Activate.ps1
```

### 2. Instale as dependências

Em um Terminal (Prompt de Comando ou PowerShell), na pasta do projeto, digite o seguinte comando:

```shell
pip install -r requirements.txt
```

### 3. Baixe o dataset

Para baixar o dataset, em um Terminal (Prompt de Comando ou PowerShell), na pasta do projeto, digite o seguinte comando:

```shell
python baixar_base_de_dados_plantvillage-dataset.py
```

### 4. Execute os modelos

Para executar o modelo ResNet50, em um Terminal (Prompt de Comando ou PowerShell), na pasta do projeto, digite o seguinte comando:

```shell
python treinar_modelo_plantvillage_resnet50.py
```

Para executar o modelo GoogleNet, em um Terminal (Prompt de Comando ou PowerShell), na pasta do projeto, digite o seguinte comando:

```shell
python treinar_modelo_plantvillage_googlenet.py
```

E para executar o modelo EfficientNetV2S, em um Terminal (Prompt de Comando ou PowerShell), na pasta do projeto, digite o seguinte comando:

```shell
python treinar_modelo_plantvillage_efficientnet.py
```

### 5. Obtendo as métricas e gerando os gráficos

Para obter as métricas do modelo ResNet50, abra a planilha [metricas_modelo_plantvillage_resnet.ipynb](metricas_modelo_plantvillage_resnet.ipynb) e execute todas as células.

Para obter as métricas do modelo GoogleNet, abra a planilha [metricas_modelo_plantvillage_googlenet.ipynb](metricas_modelo_plantvillage_googlenet.ipynb) e execute todas as células.

Para obter as métricas do modelo EfficientNetV2S, abra a planilha [metricas_modelo_plantvillage_googlenet.ipynb](metricas_modelo_plantvillage_googlenet.ipynb) e execute todas as células.
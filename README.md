# README — Cartas de Controle para monitorar a estabilidade da variável “Amount” (Transações Financeiras)
O objetivo deste projeto é aplicar o Controle Estatístico de Processo (CEP) sobre dados reais de transações financeiras, a partir do dataset Credit Card Fraud Detection (Kaggle), analisando a estabilidade da variável monetária Amount.
O sistema gera cartas de controle X̄ e R para monitorar a média e a variabilidade do processo, identificando possíveis pontos fora de controle e sugerindo investigações sobre causas especiais.

# Hipótese
A hipótese central é que os valores de Amount (montantes das transações) apresentam comportamento estável ao longo das observações, exceto quando ocorrem picos anormais possivelmente associados a transações fraudulentas ou erros de registro.
O uso das cartas de controle visa detectar pontos fora dos limites estatísticos, auxiliando na detecção de anomalias.

## 1) Configuração Inicial e Importação das Bibliotecas
```python
O código utiliza bibliotecas padrão de análise de dados e visualização, garantindo compatibilidade com o Google Colab.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")
np.random.seed(42)
print("✅ Bibliotecas importadas com sucesso!")
```
Essas bibliotecas permitem manipular dados (Pandas/Numpy), gerar gráficos (Matplotlib/Seaborn) e controlar a reprodutibilidade dos resultados.

## 2) Carregamento e Visualização do Dataset

O código baixa automaticamente o dataset público Credit Card Fraud Detection do Kaggle:


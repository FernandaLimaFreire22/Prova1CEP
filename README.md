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
```python
url = "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
print("📥 Baixando dataset do Kaggle (pode demorar alguns segundos)...")

# O arquivo principal é creditcard.csv
df = pd.read_csv("/content/data/creditcard.csv")
print("Dimensão do dataset:", df.shape)
display(df.head())
```
A base contém 284.807 registros e 31 colunas, incluindo variáveis anônimas (V1–V28), o valor da transação (Amount) e a variável alvo (Class) indicando fraude (1) ou não (0).

## 3) Seleção da Variável e Parâmetros do CEP
O código define parâmetros principais:
* MEASURE_COL = "Amount"
* USE_LOG = False
* SUBGROUP_SIZE = 5
* MAX_OBS_TO_USE = 300
Esses parâmetros indicam que serão utilizadas as 300 primeiras observações do dataset, organizadas em 60 subgrupos de tamanho 5.

## 4) Formação dos Subgrupos
Os subgrupos são formados para calcular estatísticas locais de cada grupo: média, amplitude e desvio-padrão.
```python
n = 5
max_obs = 300

values = df["Amount"].values[:max_obs]
m = len(values) // n
values_use = values[:m*n]

groups = values_use.reshape(m, n)
sub_mean = groups.mean(axis=1)
sub_range = np.ptp(groups, axis=1)
sub_std = groups.std(axis=1, ddof=1)

sub_df = pd.DataFrame({
    "subgroup": np.arange(1, m+1),
    "mean": sub_mean,
    "range": sub_range,
    "std": sub_std
})
display(sub_df.head())
```
Resultado:Foram formados 60 subgrupos de tamanho 5.
Esses subgrupos são usados para estimar os limites de controle.

## 5) Cálculo dos Limites de Controle (X̄ e R)
As constantes do CEP para subgrupos de tamanho 5 são aplicadas (A2=0.577, D3=0.000, D4=2.114, d2=2.326).
Com base nas médias e amplitudes calculadas:
```python
A2, D3, D4 = 0.577, 0.000, 2.114

Xbar_bar = sub_df['mean'].mean()
R_bar = sub_df['range'].mean()
S_bar = sub_df['std'].mean()

UCL_xbar = Xbar_bar + A2 * R_bar
LCL_xbar = Xbar_bar - A2 * R_bar
UCL_R = D4 * R_bar
LCL_R = D3 * R_bar
```
📐 Resultados gerais:
| Estatística | Valor                                 |
| ----------- | --------------------------------------|
| X̄-bar      | 85.55                                  |
| R̄          | 301.02                                 |
| S̄          | 130.86                                 |
| Limites X̄  | LCL = -88.13, CL = 85.55, UCL = 259.24 |
| Limites R   | LCL = 0.00, CL = 301.02, UCL = 636.35 |

## 6) Geração das Cartas de Controle
As cartas X̄ e R são geradas com destaque para os pontos fora de controle (em vermelho):
```python
def detect_ooc(series, lcl, ucl):
    return (series < lcl) | (series > ucl)

idx = sub_df['subgroup']

# Carta X-bar
plt.figure(figsize=(12,5))
plt.plot(idx, sub_df['mean'], 'o-', label='Média do subgrupo')
plt.axhline(Xbar_bar, color='green', linestyle='--', label='CL (X-bar)')
plt.axhline(UCL_xbar, color='red', linestyle='--', label='UCL')
plt.axhline(LCL_xbar, color='red', linestyle='--', label='LCL')

ooc_x = detect_ooc(sub_df['mean'], LCL_xbar, UCL_xbar)
plt.scatter(idx[ooc_x], sub_df['mean'][ooc_x], color='red', s=90, label='Fora de controle (X)')
plt.title("Carta X̄ para 'Amount' (n=5)")
plt.legend()
plt.show()

# Carta R
plt.figure(figsize=(12,5))
plt.plot(idx, sub_df['range'], 'o-', label='Amplitude do subgrupo (R)')
plt.axhline(R_bar, color='green', linestyle='--', label='R (CL)')
plt.axhline(UCL_R, color='red', linestyle='--', label='UCL_R')
plt.axhline(LCL_R, color='red', linestyle='--', label='LCL_R')

ooc_r = detect_ooc(sub_df['range'], LCL_R, UCL_R)
plt.scatter(idx[ooc_r], sub_df['range'][ooc_r], color='red', s=90, label='Fora de controle (R)')
plt.title("Carta R para 'Amount' (n=5)")
plt.legend()
plt.show()
```
## 7) Interpretação dos Resultados
O sistema detectou pontos fora de controle:
* X̄ fora de controle: subgrupos [11, 18, 29, 33]
* R fora de controle: subgrupos [11, 18, 29, 31, 33, 35, 49]

🔎 Análise:
* Há indícios de causas especiais em múltiplos subgrupos, com picos extremos (como o subgrupo 33).
* A variabilidade também mostra instabilidade, indicando que o processo não está sob controle estatístico.
* Recomenda-se investigar eventos específicos (horário, tipo de operação, lote, etc.) nos subgrupos anômalos.

## 8) Recomendações Técnicas
* Revisar as observações dos subgrupos fora de controle, verificando se correspondem a fraudes, erros de medição ou registros excepcionais.
* Calibrar instrumentos de coleta e checar consistência dos dados.
* Aplicar transformação logarítmica (log(Amount+1)) para reduzir assimetria e facilitar a detecção de padrões.
* Reexecutar as cartas de controle após ajustes para confirmar se o processo se estabiliza.

O projeto demonstrou a aplicação prática do Controle Estatístico de Processo (CEP) sobre dados reais.
O comportamento da variável “Amount” revelou instabilidade em média e variabilidade, exigindo investigação dos subgrupos fora de controle para entender as causas e aplicar ações corretivas.

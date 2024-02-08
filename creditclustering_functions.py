import pandas as pd
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def processamento(file):
    dados = pd.read_csv(file)
    dados = dados.drop(['CUST_ID', 'TENURE'], axis=1)
    missing = dados.isna().sum()
    print(missing)
    dados.fillna(dados.median(), inplace=True)
    
    return dados


#Qual modelo de normalização usar? 
#Função StandardScaler() --> Útil para dados que estão em diferentes escalas e possuem diferentes unidades de medida.
#Função Normalizer() --> Útil para dados que possuem escalas similares, porém possuem unidades de medidas diferentes.
#Nesse caso, deve utilizar o Normalizer().


def normalizacao(dados):
    values = Normalizer().fit_transform(dados)
    
    return values



#Métrica de precisão ----------------------- Inetria(WCSS)
def calculate_inertia(X):
    inertia_values = []

# Testar diferentes números de clusters (K) para o K-Means
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(X)
        inertia = kmeans.inertia_
        inertia_values.append(inertia)
    print('Métrica de Precisão ------ Inertia: ',inertia_values)

# Plotar um gráfico do valor de Inertia em função do número de clusters (K)
    plt.plot(range(1, 11), inertia_values, marker='o')
    plt.title('Gráfico de Inertia em função do número de clusters (K)')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Inertia (WCSS)')
    plt.show()
    
    return inertia_values


def optimal_number_of_clusters(inertia_values):
    x1, y1 = 2, inertia_values[0]
    x2, y2 = 11, inertia_values[len(inertia_values)-1]

    distances = []
    for i in range(len(inertia_values)):
        x0 = i+2
        y0 = inertia_values[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = (((y2 - y1)**2 + (x2 - x1)**2)**0.5)
        distances.append(numerator/denominator)
    
    return distances.index(max(distances)) + 2


def clustering(number_optimal, values, dados):
    kmeans = KMeans(n_clusters=number_optimal, n_init=10, max_iter=300) # Definindo o número de clusters, irá variar de 0 até 4.
    y_pred = kmeans.fit_predict(values) #Previsão da segmentação de mercado
    y_pred = pd.DataFrame(y_pred)
    dados['Cluster'] = y_pred

    return kmeans, dados['Cluster']
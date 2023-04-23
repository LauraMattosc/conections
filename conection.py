import streamlit as st
from faker import Faker
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import plotly.graph_objs as go
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from streamlit_agraph import st_agraph


# Gerar dados falsos
fake = Faker('pt_BR')
themes = ['Educação', 'Saúde', 'Meio Ambiente', 'Finanças', 'Tecnologia', 'Direito', 'Marketing', 'Logística']

def create_fake_data(n):
    data = []
    for _ in range(n):
        name = fake.name()
        theme = random.choice(themes)
        data.append((name, theme))
    return data

def encode_data(data):
    encoder = OneHotEncoder()
    encoded_data = encoder.fit_transform(data).toarray()
    return encoded_data

def cluster_data(data):
    n_clusters = len(themes)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(data)
    return clusters

def reduce_dimensionality(data):
    tsne = TSNE(n_components=2, random_state=42)
    reduced_data = tsne.fit_transform(data)
    return reduced_data

def separate_points(data, clusters):
    cluster_points = {i: ([], []) for i in range(len(themes))}
    for i, point in enumerate(data):
        cluster = clusters[i]
        cluster_points[cluster][0].append(point[0])
        cluster_points[cluster][1].append(point[1])
    return cluster_points

# Criar o aplicativo Streamlit
st.title('Clusters de Pessoas por Temática')
st.write('Este aplicativo gera dados falsos de pessoas com temas aleatórios e os agrupa em clusters usando k-means e os visualiza em um gráfico interativo.')

# Definir o número de pontos de dados
n_points = st.slider('Selecione o número de pontos de dados a serem gerados', min_value=100, max_value=700, step=100, value=100)

# Gerar dados falsos
fake_data = create_fake_data(n_points)

# Codificar os dados categóricos
encoded_data = encode_data(fake_data)

# Agrupar dados
clusters = cluster_data(encoded_data)

# Reduzir a dimensionalidade dos dados para 2D usando t-SNE
reduced_data = reduce_dimensionality(encoded_data)

# Separar pontos por cluster
cluster_points = separate_points(reduced_data, clusters)


# Criar nós (pessoas) do gráfico de rede
nodes = []
for i, person in enumerate(fake_data):
    nodes.append((i, {'label': person[0], 'theme': person[1]}))

# Criar conexões do gráfico de rede
edges = []
for i, person in enumerate(fake_data):
    recommendations = get_recommendations(i, encoded_data)
    for rec in recommendations:
        edges.append((i, rec))

# Criar o grafo de rede
graph = st.agraph.graph(name='Recomendações de pessoas por similaridade temática')

# Adicionar nós (pessoas) ao grafo de rede
for node in nodes:
    graph.add_node(name=str(node[0]), label=node[1]['label'], shape='circle', style='filled',
                   fillcolor=colors[themes.index(node[1]['theme'])], fontcolor='white', fontsize=10)

# Adicionar conexões ao grafo de rede
for edge in edges:
    graph.add_edge(str(edge[0]), str(edge[1]))

# Definir configurações do grafo de rede
graph.node_attr.update(fontname='Helvetica', fontcolor='black')
graph.edge_attr.update(color='gray', arrowsize=0.8)

# Exibir grafo de rede
st.graphviz_chart(graph)

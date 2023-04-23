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
import numpy as np



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
n_points = st.slider('Selecione o número de pontos de dados a serem gerados', min_value=100, max_value=200, step=100, value=100)

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

# Criar um gráfico interativo com plotly
fig = go.Figure()

colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'purple']

for i, points in cluster_points.items():
    fig.add_trace(go.Scatter(x=points[0], y=points[1], mode='markers',
                             marker=dict(color=colors[i], size=8),
                             text=[f"{fake_data[j][0]}<br>{themes[i]}" for j in range(len(fake_data)) if clusters[j] == i],
                             name=themes[i]))

# Exibir o gráfico interativo
st.plotly_chart(fig, use_container_width=True)

def get_recommendations(person_index, encoded_data):
    dist_matrix = distance_matrix(encoded_data, encoded_data)
    sorted_indices = np.argsort(dist_matrix[person_index])
    similar_recommendations = sorted_indices[1:4]
    different_recommendations = sorted_indices[-3:]
    return np.concatenate((similar_recommendations, different_recommendations))

# Criar um grafo vazio
G = nx.Graph()

# Adicionar nós (pessoas) ao grafo
for i, person in enumerate(fake_data):
    G.add_node(i, label=person[0], theme=person[1])

# Adicionar arestas (conexões) ao grafo
for i, person in enumerate(fake_data):
    recommendations = get_recommendations(i, encoded_data)
    for rec in recommendations:
        G.add_edge(i, rec)

# Desenhar o grafo de rede
fig, ax = plt.subplots(figsize=(12, 12))
pos = nx.spring_layout(G, seed=42, iterations=50, cooling=0.95)
nx.draw(G, pos, node_color=[colors[themes.index(G.nodes[node]["theme"])] for node in G], with_labels=False, ax=ax)
nx.draw_networkx_labels(G, pos, labels={node: G.nodes[node]["label"] for node in G}, font_size=8, ax=ax)

# Exibir gráfico
st.pyplot(fig)

import streamlit as st
from faker import Faker
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import plotly.graph_objs as go
import numpy as np
from scipy.spatial import distance_matrix
import networkx as nx
import matplotlib.pyplot as plt

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

fake_data = create_fake_data(100)

# Codificar dados categóricos
encoder = OneHotEncoder()
encoded_data = encoder.fit_transform(fake_data).toarray()

# Agrupar dados
n_clusters = len(themes) # Um cluster para cada tema
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
clusters = kmeans.fit_predict(encoded_data)

# Reduzir a dimensionalidade para 2D usando t-SNE
tsne = TSNE(n_components=2, random_state=42)
reduced_data = tsne.fit_transform(encoded_data)

# Separar pontos por cluster
cluster_points = {i: ([], []) for i in range(n_clusters)}
for i, point in enumerate(reduced_data):
    cluster = clusters[i]
    cluster_points[cluster][0].append(point[0])
    cluster_points[cluster][1].append(point[1])

# Criar um gráfico interativo com plotly
fig = go.Figure()

colors = ['red', 'green', 'blue', 'cyan', 'magenta', 'yellow', 'black', 'purple']

for i, points in cluster_points.items():
    fig.add_trace(go.Scatter(x=points[0], y=points[1], mode='markers',
                             marker=dict(color=colors[i], size=8),
                             text=[f"{fake_data[j][0]}<br>{themes[i]}" for j in range(len(fake_data)) if clusters[j] == i],
                             name=themes[i]))

fig.update_layout(title='Clusters de Pessoas por Temática', hovermode='closest')

st.title("Clusters de Pessoas por Temática")
st.plotly_chart(fig)

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
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(G, seed=42, iterations=50)
nx.draw(G, pos, node_color=[colors[themes.index(G.nodes[node]["theme"])] for node in G], with_labels=False)
nx.draw_networkx_labels(G, pos, labels={node: G.nodes[node]["label"] for node in G}, font_size=8)

# Exibir gráfico
st.title("Rede de Conexões Recomendadas")
st.pyplot(plt.gcf())
       

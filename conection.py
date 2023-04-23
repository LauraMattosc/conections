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
from pyvis.network import Network

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

# Criar um gráfico interativo com pyvis
net = Network(height='600px', width='100%', bgcolor='#222222', font_color='white', directed=False)
net.barnes_hut()
for i, theme in enumerate(themes):
    net.add_node(i, label=theme, color=colors[i])

for i, points in cluster_points.items():
    for j in range(len(fake_data)):
        if clusters[j] == i:
            net.add_node(j+len(themes), label=fake_data[j][0], color=colors[i])
            net.add_edge(i, j+len(themes))

# Criar filtros
theme_filter = st.sidebar.selectbox("Filtrar por tema", themes)
filtered_nodes = [i+len(themes) for i in range(len(fake_data)) if fake_data[i][1] == theme_filter]
if len(filtered_nodes) > 0:
    net.set_options(f"{theme_filter} = true;")
    net.toggle_physics(False)
    net.highlight_nodes(filtered_nodes)
else:
    net.toggle_physics(True)

# Exibir gráfico
st.title("Rede de Conexões Recomendadas")
st_pyvis(net)    
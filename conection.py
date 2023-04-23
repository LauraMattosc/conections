import streamlit as st
from faker import Faker
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
from pyvis.network import Network
import numpy as np

# Definir temas e cores
themes = ['Educação', 'Saúde', 'Meio Ambiente', 'Finanças', 'Tecnologia', 'Direito', 'Marketing', 'Logística']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

def create_fake_data(n):
    fake = Faker('pt_BR')
    data = []
    for i in range(n):
        name = fake.name()
        theme = random.choice(themes)
        data.append((i, name, theme))
    return data

fake_data = create_fake_data(100)

# Codificar dados categóricos
encoder = OneHotEncoder()
encoded_data = encoder.fit_transform([(x[2],) for x in fake_data]).toarray()

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
            net.add_node(fake_data[j][0], label=fake_data[j][1], title=fake_data[j][2], color=colors[i])
            net.add_edge(i, fake_data[j][0])
            
# Exibir gráfico
st.title("Rede de Conexões Recomendadas")
st_pyvis(net)


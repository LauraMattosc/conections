import streamlit as st
from faker import Faker
import random
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import numpy as np
import networkx as nx
from pyvis.network import Network
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Definir temas e cores
themes = ['Educação', 'Saúde', 'Meio Ambiente', 'Finanças', 'Tecnologia', 'Direito', 'Marketing', 'Logística']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']

@st.cache_data()
def create_fake_data(n):
    fake = Faker('pt_BR')
    data = []
    for _ in range(n):
        name = fake.name()
        theme = random.choice(themes)
        job = fake.job()
        data.append((name, theme, job))
    return data

fake_data = create_fake_data(100)

# Codificar dados categóricos
encoder = OneHotEncoder()
encoded_data = encoder.fit_transform([[x[1]] for x in fake_data]).toarray()

# Agrupar dados
n_clusters = len(themes) # Um cluster para cada tema
kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
clusters = kmeans.fit_predict(encoded_data)

# Reduzir a dimensionalidade para 2D usando t-SNE
tsne = TSNE(n_components=2, random_state=42)
reduced_data = tsne.fit_transform(encoded_data)

# Criar um grafo com NetworkX
G = nx.Graph()

# Adicionar nós ao grafo
for i, data in enumerate(fake_data):
    G.add_node(i, name=data[0], theme=data[1], job=data[2], cluster=clusters[i])

# Adicionar arestas ao grafo
for i in range(len(fake_data)):
    for j in range(i+1, len(fake_data)):
        if clusters[i] == clusters[j]:
            G.add_edge(i, j)

# Criar um grafo interativo com pyvis
net = Network(height='600px', width='100%', bgcolor='#222222', font_color='white', directed=False, notebook=False)

# Adicionar nós e arestas ao grafo
for node in G.nodes:
    net.add_node(node, label=f"{G.nodes[node]['name']} ({G.nodes[node]['job']})", title=G.nodes[node]['theme'], color=colors[G.nodes[node]['cluster']])
for edge in G.edges:
    net.add_edge(edge[0], edge[1])

# Exibir grafo
st.title("Rede de Conexões Recomendadas")
net.show("network.html")
HtmlFile = open("network.html", 'r', encoding='utf-8')
source_code = HtmlFile.read()
components.html(source_code, height=600) 

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

def separate_points(reduced_data, clusters):
    cluster_points = {i: ([], []) for i in range(len(themes))}
    for i, point in enumerate(reduced_data):
        cluster = clusters[i]
        cluster_points[cluster][0].append(point[0])
        cluster_points[cluster][1].append(point[1])
    return cluster_points

# Obter recomendações de pessoas por similaridade temática
def get_recommendations(person_index, data):
    dist_matrix = distance_matrix(data, data)
    similar_indices = dist_matrix[person_index].argsort()[1:6]
    return similar_indices

# Criar o aplicativo Streamlit
st.title('Clusters de Pessoas por Temática')
st.write('Este aplicativo gera dados falsos de pessoas com temas aleatórios e os agrupa em clusters usando k-means e os visualiza em um gráfico interativo.')

# Definir o número de pontos de dados
n_points = st.slider('Selecione o número de pontos de dados a serem gerados', min_value=100, max_value=700, step=100, value=100)

# Gerar dados falsos
fake_data = create_fake_data(n_points)

# Codificar os dados categóricos
encoded_data = encode_data(fake_data)

# Reduzir a dimensionalidade dos dados para 2D usando t-SNE
reduced_data = reduce_dimensionality(encoded_data)

# Agrupar dados
clusters = cluster_data(encoded_data)

# Separar pontos por cluster
cluster_points = separate_points(reduced_data, clusters)

fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(title='Recomendações de pessoas por similaridade temática', showlegend=False,
                                 hovermode='closest', margin=dict(b=20, l=5, r=5, t=40),
                                 xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                 yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))

# Exibir o gráfico interativo
st.plotly_chart(fig, use_container_width=True)






# Criar o grafo de rede
graph = nx.Graph(name='Recomendações de pessoas por similaridade temática')

# Adicionar nós (pessoas) ao grafo de rede
for i, person in enumerate(fake_data):
    graph.add_node(i, label=person[0], theme=person[1], shape='circle', style='filled',
                   fillcolor=colors[themes.index(person[1])], fontcolor='white', fontsize=10)

# Adicionar conexões ao grafo de rede
for i, person in enumerate(fake_data):
    recommendations = get_recommendations(i, encoded_data)
    for rec in recommendations:
        graph.add_edge(i, rec)

# Definir configurações do grafo de rede
pos = nx.spring_layout(graph, seed=42)
edge_x = []
edge_y = []
for edge in graph.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    edge_x.append(x0)
    edge_x.append(x1)
    edge_x.append(None)
    edge_y.append(y0)
    edge_y.append(y1)
    edge_y.append(None)

node_x = []
node_y = []
for node in graph.nodes():
    x, y = pos[node]
    node_x.append(x)
    node_y.append(y)

node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', text=[person[0] for person in fake_data],
                        marker=dict(color=[colors[themes.index(person[1])] for person in fake_data], size=10))

edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='gray'), hoverinfo='none')


# Exibir grafo de rede
st.plotly_chart(fig, use_container_width=True)

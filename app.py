import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import matplotlib.pyplot as plt
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import zscore
pd.options.mode.chained_assignment = None  
import numpy as np
import plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

st.title('NBA DuoFit')
st.write("Aren't you curious to find out hypothetically how would 2 NBA AllStars Fit Together?")
st.write('PS: I built this due to my curiosity after I heard about the Bradley Beal to the Suns trade')


combined = pd.read_csv('combined.csv')
combined = combined.sort_values(by='value_over_replacement_player', ascending=False).drop_duplicates(
    keep='first', subset=['name']) 
combined = combined.reset_index(drop=True)
combined['positions'] = combined['positions'].astype('category')
combined['team'] = combined['team'].astype('category')
data = [
    "LeBron James", "Dwyane Wade", "Chris Bosh", "Paul George", "Carmelo Anthony",
    "Kyrie Irving", "Kevin Durant", "Blake Griffin", "Kevin Love", "Stephen Curry",
    "James Harden", "Tony Parker", "LaMarcus Aldridge", "Dirk Nowitzki", "John Wall",
    "DeMar DeRozan", "Joakim Noah", "Paul Millsap", "Al Horford", "Chris Paul",
    "Anthony Davis", "Damian Lillard", "Klay Thompson", "Russell Westbrook",
    "DeMarcus Cousins", "Marc Gasol", "Tim Duncan", "Zach Randolph", "Giannis Antetokounmpo",
    "DeAndre Jordan", "Draymond Green", "Kemba Walker", "Gordon Hayward", "Victor Oladipo",
    "Karl-Anthony Towns", "Ben Simmons", "Donovan Mitchell",
    "Rudy Gobert", "Zion Williamson", "Nikola Jokić", "Kawhi Leonard", "Devin Booker",
    "Domantas Sabonis", "Trae Young", "Jayson Tatum", "Jaylen Brown", "De'Aaron Fox",
    "Luka Dončić","Fred VanVleet","Darius Garland","Jimmy Butler","Jarrett Allen","Ja Morant","Andrew Wiggins",
    "Joel Embiid","LaMelo Ball","Dejounte Murray","Zach LaVine","Khris Middleton","Lauri Markkanen","Bam Adebayo",
    "Jrue Holiday","Domantas Sabonis","Pascal Siakam","Anthony Edwards","De'Aaron Fox",
    "Tyrese Haliburton","Julius Randle",'Jaren Jackson Jr.', 'Kristaps Porziņģis', 'Shai Gilgeous-Alexander']
data = pd.Series(data)
allstars = combined[combined['name'].isin(data)]
allstars = allstars.reset_index()
normdf = allstars.loc[:,'player_efficiency_rating':'value_over_replacement_player'].apply(zscore)
only_nums=normdf
min_max_scaler = MinMaxScaler()
scaled_data_minmax = min_max_scaler.fit_transform(only_nums)
only_nums = pd.DataFrame(scaled_data_minmax, columns=only_nums.columns)
normdf['name'] = allstars['name']
normdf['positions']=allstars['positions']
normdf['team']=allstars['team']
cols = list(normdf.columns)
cols = cols[-3:] + cols[:-3]
normdf = normdf[cols]
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(only_nums)
allstars['Cluster'] = labels

pca_3d = PCA(n_components=3)
normdf['Cluster'] = allstars['Cluster']
PCs_3d = pd.DataFrame(pca_3d.fit_transform(normdf.drop(['Cluster','name','positions','team'],axis=1)))
PCs_3d.columns = ["PC1_3d", "PC2_3d", "PC3_3d"]
plotX = pd.concat([normdf,PCs_3d], axis=1, join='inner')
cluster0 = plotX[plotX["Cluster"] == 0]
cluster1 = plotX[plotX["Cluster"] == 1]
cluster2 = plotX[plotX["Cluster"] == 2]
init_notebook_mode(connected=True)
hover_text_cluster0 = [f"Player: {name}" for name in cluster0["name"]]
hover_text_cluster1 = [f"Player: {name}" for name in cluster1["name"]]
hover_text_cluster2 = [f"Player: {name}" for name in cluster2["name"]]

trace1 = go.Scatter3d(
                    x = cluster0["PC1_3d"],
                    y = cluster0["PC2_3d"],
                    z = cluster0["PC3_3d"],
                    mode = "markers",
                    name = "Cluster 0",
                    marker = dict(color = 'blue'),
                    text = hover_text_cluster0)
trace2 = go.Scatter3d(
                    x = cluster1["PC1_3d"],
                    y = cluster1["PC2_3d"],
                    z = cluster1["PC3_3d"],
                    mode = "markers",
                    name = "Cluster 1",
                    marker = dict(color = 'red'),
                    text = hover_text_cluster1)
trace3 = go.Scatter3d(
                    x = cluster2["PC1_3d"],
                    y = cluster2["PC2_3d"],
                    z = cluster2["PC3_3d"],
                    mode = "markers",
                    name = "Cluster 2",
                    marker = dict(color = 'green'),
                    text = hover_text_cluster2)

plotter = [trace1, trace2, trace3]

title = "Visualizing AllStar Clusters"

layout = dict(title = title,
              xaxis= dict(title= 'PC1',ticklen= 5,zeroline= False),
              yaxis= dict(title= 'PC2',ticklen= 5,zeroline= False)
             )
fig = dict(data = plotter, layout = layout)

option1 = st.selectbox(
    'Select Player 1', data.values, placeholder = 'None Selected')
option2 = st.selectbox(
    'Select Player 2', data.values, placeholder = 'None Selected')

duos = pd.read_excel('duos.xlsx')
duos.columns = duos.iloc[0]
duos = duos[1:]
duos['LINEUPS']=duos['LINEUPS'].astype(str)
duos[['Player1', 'Player2']] = duos['LINEUPS'].str.rsplit('-', n=1, expand=True) 
duos.drop(['LINEUPS'],inplace=True,axis=1)

matched_rows = []
for index, row in duos.iterrows():
    player1 = row['Player1']
    player2 = row['Player2']
    netrtng = row['NETRTG']
    
    match_count = 0
    
    for full_name in data.values:
        similarity1 = fuzz.token_set_ratio(player1, full_name)
        similarity2 = fuzz.token_set_ratio(player2, full_name)
        
        if similarity1 >= 80:
            match_count += 1
        
        if similarity2 >= 80:
            match_count += 1
    
    if match_count >= 2:  # Change this threshold if necessary
        matched_rows.append((player1, player2,netrtng))
matched_duos = pd.DataFrame(matched_rows, columns=['Player1', 'Player2','Net_Rating'])

PCs_3d['name'] = normdf['name'].reset_index().drop('index',axis=1).values
selected_player1 = option1
selected_player2 = option2
# Retrieve PCA representations of selected players using PCA dataset
selected_player1_row = PCs_3d[PCs_3d['name'] == selected_player1].iloc[0]
selected_player1_pca = selected_player1_row[['PC1_3d', 'PC2_3d', 'PC3_3d']]

selected_player2_row = PCs_3d[PCs_3d['name'] == selected_player2].iloc[0]
selected_player2_pca = selected_player2_row[['PC1_3d', 'PC2_3d', 'PC3_3d']]

# Calculate Euclidean distances for each player in the dataset
distances = []
for index, player_row in PCs_3d.iterrows():
    player_name = player_row['name']
    player_pca = player_row[['PC1_3d', 'PC2_3d', 'PC3_3d']]
    
    distance1 = np.linalg.norm(selected_player1_pca - player_pca)
    distance2 = np.linalg.norm(selected_player2_pca - player_pca)
    
    distances.append((player_name, distance1, distance2))

# Sort players based on distance1 
distances.sort(key=lambda x: x[1])

# Find the 20 closest players for selected_player1
closest_players1 = [player[0] for player in distances[:20]]

# Sort players based on distance2
distances.sort(key=lambda x: x[2])

# Find the 20 closest players for selected_player2
closest_players2 = [player[0] for player in distances[:20]]


names = np.unique(np.append(closest_players1,closest_players2))
matched_rows = []
for index, row in matched_duos.iterrows():
    player1 = row['Player1']
    player2 = row['Player2']
    netrtng = row['Net_Rating']
    
    match_count = 0
    
    for full_name in names:
        similarity1 = fuzz.token_set_ratio(player1, full_name)
        similarity2 = fuzz.token_set_ratio(player2, full_name)
        
        if similarity1 >= 80:
            match_count += 1
        
        if similarity2 >= 80:
            match_count += 1
    
    if match_count >= 2:  
        matched_rows.append((player1, player2,netrtng))

filtered_df = pd.DataFrame(matched_rows, columns=["player1", "player2",'netrating'])
final = (filtered_df['netrating'].mean())/matched_duos['Net_Rating'].mean()

if 0<=final<=1:
    formatted_text = f'<span style="color: red;">{final}</span>'
    full_text = f"Net Rating Ratio Estimate: {formatted_text}"
    st.write(full_text, unsafe_allow_html=True)
if 1<final<=1.5:
    formatted_text = f'<span style="color: yellow;">{final}</span>'
    full_text = f"Net Rating Ratio Estimate: {formatted_text}"
    st.write(full_text, unsafe_allow_html=True)
if final >1.5:
    formatted_text = f'<span style="color: green;">{final}</span>'
    full_text = f"Net Rating Ratio Estimate: {formatted_text}"
    st.write(full_text, unsafe_allow_html=True)

st.header('Algorithm & Logic:')
list_items = [
    "Condense NBA players into 3 dimensional spaces using Principal Component Analysis. Think of it as condensing their 30 statistical measures into 3 to be able to plot them in an xyz plane",
    "Find the 20 most similar players to each selected player, and find the actual average net rating for any existng combinations of those players (using 2 man NBA lineup data)",
    "Check how this net rating compares to the average 2 man NBA all star net rating",
]
numbered_list = "<ol>" + "".join([f"<li>{item}</li>" for item in list_items]) + "</ol>"
st.write(numbered_list, unsafe_allow_html=True)

st.markdown('**:red[Think about if you were to visualize how far two points are on an xy graph. Simple. How about if I asked how similar/dissimilar are 2 NBA All Stars?]**')
st.markdown("Hover over any point to see which player it is")
st.plotly_chart(fig)

st.write(pd.DataFrame({'Closest1': closest_players1, 'Closest2': closest_players2}))

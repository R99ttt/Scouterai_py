from db.database import get_engine, get_session
from db.models import Player
from ml.train import train_model
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE

def main():
    engine = get_engine()
    session = get_session(engine)

    try:
        print("Querying the database for players in FIFA version 23...")

        start_time = time.time()
        players = session.query(
            Player.age.label('Age'),
            Player.overall.label('Overall'),
            Player.potential.label('Potential'),
            Player.attacking_crossing.label('Crossing'),
            Player.attacking_finishing.label('Finishing'),
            Player.attacking_heading_accuracy.label('HeadingAccuracy'),
            Player.attacking_short_passing.label('ShortPassing'),
            Player.attacking_volleys.label('Volleys'),
            Player.skill_dribbling.label('Dribbling'),
            Player.skill_curve.label('Curve'),
            Player.mentality_penalties.label('Penalties'),
            Player.mentality_composure.label('Composure'),
            Player.defending_marking_awareness.label('Marking'),
            Player.defending_standing_tackle.label('StandingTackle'),
            Player.defending_sliding_tackle.label('SlidingTackle'),
            Player.goalkeeping_diving.label('GKDiving'),
            Player.goalkeeping_handling.label('GKHandling'),
            Player.goalkeeping_kicking.label('GKKicking'),
            Player.goalkeeping_positioning.label('GKPositioning'),
            Player.goalkeeping_reflexes.label('GKReflexes')
        ).filter(Player.fifa_version=='23').all()

        end_time = time.time()

        query_time = end_time - start_time
        print(f"Query completed in {query_time:.2f} seconds.")

        df_full = pd.DataFrame(players, columns=[
            'Age', 'Overall', 'Potential', 'Crossing', 'Finishing', 'HeadingAccuracy', 
            'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'Penalties', 
            'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 
            'GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes'
        ])


        if not players:
            print("No players found!")
        else:
            print(f"Found {len(players)} players in FIFA version 23.")
            df_clean = df_full.dropna()  # Drop rows with missing values

            # PCA with 1 component
            pca = PCA(n_components=1).fit(df_clean)
            explained_variance = round(float(pca.explained_variance_ratio_), 3)
            print("Explained variance by 1st principal component:", explained_variance)

            # PCA to retain 95% variance
            pca_095 = PCA(n_components=0.95)
            X_reduced = pca_095.fit_transform(df_clean)
            num_components = X_reduced.shape[1]
            print("Number of components to retain 95% variance:", num_components)

            # PCA with 2 components
            # pca_w2 = PCA(n_components=2)
            # pca_w2.fit(df_clean)

            # RFE for feature selection
            x = df_clean.drop('Potential', axis=1)  # Drop the target column 'Potential'
            y = df_clean['Potential']  # Target column
            reg = LinearRegression().fit(x, y)
            rfe = RFE(reg, n_features_to_select=5).fit(x, y)
            nom_var = x.loc[:, rfe.get_support()].columns
            selected_features = list(nom_var)
            print("Selected features:", selected_features)

    except Exception as e:
        print("An error occurred during the query or session.")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

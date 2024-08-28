import pickle
import pandas as pd
from sqlalchemy import create_engine, text
import numpy as np

# Load models, scalers, and label encoder
with open('overallModel.pkl', 'rb') as f:
    overall_model = pickle.load(f)
with open('potentialModel.pkl', 'rb') as f:
    potential_model = pickle.load(f)
with open('overallScaler.pkl', 'rb') as f:
    overall_scaler = pickle.load(f)
with open('potentialScaler.pkl', 'rb') as f:
    potential_scaler = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Create a database connection using SQLAlchemy for better performance with pandas
engine = create_engine('mysql+mysqlconnector://root:password@localhost/player_data_db', echo=True)

# Fetch data in chunks to manage memory and CPU load
chunk_size = 10000  # Adjust this based on your system's capacity

query = """
    SELECT pr.id, pr.overall, pr.pace, pr.shooting, pr.passing, pr.dribbling, 
           pr.defending, pr.physic, pr.attacking_crossing, pr.attacking_short_passing, 
           pr.goalkeeping_positioning, pr.goalkeeping_diving, pr.goalkeeping_handling, 
           pr.goalkeeping_kicking, pr.goalkeeping_reflexes, pr.goalkeeping_speed, 
           pi.international_reputation, pa.age, pa.skill_moves,
           pi.player_positions
    FROM playerRatings pr
    JOIN playerIterations pi ON pr.player_iteration_id = pi.id
    JOIN playerAttributes pa ON pi.id = pa.player_iteration_id
"""

# Process data in chunks
for chunk in pd.read_sql_query(query, engine, chunksize=chunk_size):
    # Identify goalkeepers based on the 'player_positions' field containing 'gk'
    chunk['is_goalkeeper'] = chunk['player_positions'].str.contains('gk', case=False, na=False)
    
    # Separate goalkeepers and outfield players
    goalkeepers = chunk[chunk['is_goalkeeper']]
    outfield_players = chunk[~chunk['is_goalkeeper']]
    
    # Handle outfield players (drop rows with NaN values)
    outfield_players.dropna(inplace=True)
    
    # Define columns to normalize
    columns_to_normalize = ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic', 'attacking_crossing', 'attacking_short_passing']
    
    # Goalkeeping attributes and weights for filling NaNs
    goalkeeping_attributes = ['goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking', 
                              'goalkeeping_positioning', 'goalkeeping_reflexes', 'goalkeeping_speed']
    
    # Assign weights based on assumed relevance (you can adjust these weights based on domain knowledge)
    weights = {
        'pace': [0.1, 0.1, 0.2, 0.4, 0.1, 0.1],
        'shooting': [0.1, 0.2, 0.2, 0.2, 0.2, 0.1],
        'passing': [0.2, 0.2, 0.1, 0.3, 0.1, 0.1],
        'dribbling': [0.2, 0.1, 0.2, 0.2, 0.2, 0.1],
        'defending': [0.1, 0.2, 0.1, 0.3, 0.2, 0.1],
        'physic': [0.2, 0.1, 0.2, 0.2, 0.1, 0.2],
        'attacking_crossing': [0.2, 0.2, 0.2, 0.2, 0.1, 0.1],
        'attacking_short_passing': [0.2, 0.2, 0.2, 0.2, 0.1, 0.1]
    }
    
    # Fill NaN values for each column based on a weighted average of goalkeeping attributes
    for col in columns_to_normalize:
        def fill_na_based_on_weighted_goalkeeping(row):
            if pd.isna(row[col]):
                available_attributes = row[goalkeeping_attributes].dropna()
                if not available_attributes.empty:
                    relevant_weights = np.array([weights[col][i] for i in range(len(goalkeeping_attributes)) if pd.notna(row[goalkeeping_attributes[i]])])
                    if relevant_weights.sum() == 0:
                        return np.nan
                    return np.average(available_attributes, weights=relevant_weights)
                else:
                    return row[col]  # Leave as NaN if no attributes are available
            return row[col]
        
        # Apply the function to fill NaN values
        goalkeepers[col] = goalkeepers.apply(fill_na_based_on_weighted_goalkeeping, axis=1)
    
    # Combine both dataframes back together
    chunk = pd.concat([goalkeepers, outfield_players])
    
    # Ensure the feature names match exactly with those used during scaler fitting
    overall_data = chunk.rename(columns={
        'age': 'Age',
        'international_reputation': 'International Reputation',
        'dribbling': 'Dribbling',
        'skill_moves': 'Skill Moves',
        'pace': 'Pace',
        'shooting': 'Shooting',
        'passing': 'Passing',
        'defending': 'Defending',
        'physic': 'Physic'
    })[['Age', 'International Reputation', 'Dribbling', 'Skill Moves', 
        'Pace', 'Shooting', 'Passing', 'Defending', 'Physic']]
    
    potential_data = chunk.rename(columns={
        'overall': 'Overall',
        'age': 'Age',
        'attacking_crossing': 'Crossing',
        'attacking_short_passing': 'ShortPassing',
        'goalkeeping_positioning': 'GKPositioning'
    })[['Overall', 'Age', 'Crossing', 'ShortPassing', 'GKPositioning']]
    
    # Scale the features
    scaled_overall_features = overall_scaler.transform(overall_data)
    scaled_potential_features = potential_scaler.transform(potential_data)
    
    # Predict using the models
    chunk['model_overall'] = overall_model.predict(scaled_overall_features).astype(int)
    
    # Predict potential categories and decode the labels
    potential_predictions_encoded = potential_model.predict(scaled_potential_features)
    chunk['model_potential'] = label_encoder.inverse_transform(potential_predictions_encoded)
    
    # Prepare data for batch update
    update_data = list(chunk[['model_overall', 'model_potential', 'id']].itertuples(index=False, name=None))

    update_query = text("""
        UPDATE playerRatings
        SET model_overall = :model_overall, model_potential = :model_potential
        WHERE id = :id
    """)
    
    with engine.connect() as connection:
        trans = connection.begin()
        for data in update_data:
            result = connection.execute(update_query, {"model_overall": data[0], "model_potential": data[1], "id": data[2]})
        trans.commit()

# Close the database connection
engine.dispose()

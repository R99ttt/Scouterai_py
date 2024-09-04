import pickle
import pandas as pd
from sqlalchemy import create_engine, text

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
           pr.goalkeeping_positioning, pi.international_reputation, pa.age, pa.skill_moves
    FROM playerRatings pr
    JOIN playerIterations pi ON pr.player_iteration_id = pi.id
    JOIN playerAttributes pa ON pi.id = pa.player_iteration_id
"""

# Process data in chunks
for chunk in pd.read_sql_query(query, engine, chunksize=chunk_size):
    # Drop rows with any NaN values
    initial_count = len(chunk)
    chunk.dropna(inplace=True)
    after_drop_count = len(chunk)
    
    if chunk.empty:
        print(f"Chunk skipped: {initial_count} rows initially, all dropped due to NaNs.")
        continue
    
    print(f"Processing chunk: {initial_count} rows initially, {after_drop_count} rows after dropping NaNs.")
    
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

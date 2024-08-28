import pickle
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from db.database import get_engine, get_session
from db.models import Player

def train_and_save_model():
    engine = get_engine()
    session = get_session(engine)

    players = session.query(
        Player.overall.label('Overall'),
        Player.age.label('Age'),
        Player.international_reputation.label('International Reputation'),
        Player.skill_dribbling.label('Dribbling'),
        Player.skill_moves.label('Skill Moves'),
        Player.pace.label('Pace'),
        Player.shooting.label('Shooting'),
        Player.passing.label('Passing'),
        Player.defending.label('Defending'),
        Player.physic.label('Physic'),
        Player.preferred_foot.label('Preferred Foot'),
    ).all()

    df = pd.DataFrame(players, columns=['Overall', 'Age', 'International Reputation',
                                        'Dribbling', 'Skill Moves', 'Pace',
                                        'Shooting', 'Passing', 'Defending',
                                        'Physic', 'Preferred Foot']) 
        

    features = ['Age', 'International Reputation',
                'Dribbling', 'Skill Moves', 'Pace',
                'Shooting', 'Passing', 'Defending',
                'Physic']

    # Target variable
    target = 'Overall'

    # Drop rows with any missing values in the specified columns
    cleaned_df = df[features + [target]].dropna()

    # Scale the features
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(cleaned_df[features])
    scaled_df = pd.DataFrame(scaled_features, columns=features)

    x = scaled_df
    y = cleaned_df[target]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Train the model
    xgb_reg = XGBRegressor(n_estimators=1000, verbosity=1, max_depth=5,
                           gamma=0.2, subsample=0.2, learning_rate=0.3)
    xgb_reg.fit(x_train, y_train)

    # Save the model and the scaler
    with open('overallModel.pkl', 'wb') as f:
        pickle.dump(xgb_reg, f)
    with open('overallScaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

if __name__ == "__main__":
    train_and_save_model()

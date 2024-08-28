import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pandas as pd
from db.database import get_engine, get_session
from db.models import Player

# Step 1: Pull the data from the database
engine = get_engine()
session = get_session(engine)

players = session.query(
    Player.potential.label('Potential'),
    Player.overall.label('Overall'),
    Player.age.label('Age'),
    Player.attacking_crossing.label('Crossing'),
    Player.attacking_short_passing.label('ShortPassing'),
    Player.goalkeeping_positioning.label('GKPositioning'),
).filter(Player.fifa_version == '23').all()

# Convert the query result to a DataFrame
df = pd.DataFrame(players, columns=['Potential', 'Overall', 'Age', 'Crossing', 'ShortPassing', 'GKPositioning'])

# Step 2: Define the potential range categories
def assign_potential_category(potential):
    if potential > 80:
        return 'Great'
    elif potential > 70:
        return 'Good'
    elif potential > 60:
        return 'Medium'
    else:
        return 'Low'

df['PotentialRange'] = df['Potential'].apply(assign_potential_category)

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(df['PotentialRange'])

# Step 3: Select features and target variable
features = ['Overall', 'Age', 'Crossing', 'ShortPassing', 'GKPositioning']
X = df[features]

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Step 5: Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 6: Train the SVM model with RBF kernel
potential_model = SVC(kernel='rbf', probability=True, C=1, gamma='scale', random_state=42)
potential_model.fit(X_train, y_train)

# Save the trained model and the scaler
with open('potential_model.pkl', 'wb') as f:
    pickle.dump(potential_model, f)

with open('potential_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

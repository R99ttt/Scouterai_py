from db.database import get_engine, get_session
from db.models import Player
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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
svm_model = SVC(kernel='rbf', probability=True, C=1, gamma='scale', random_state=42)
svm_model.fit(X_train, y_train)

# Evaluate the SVM model
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with SVM RBF: {accuracy:.4f}")
print(f"Classification Report with SVM RBF:\n{classification_report(y_test, y_pred)}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC-AUC Score (for binary classification, use only if applicable)
if len(label_encoder.classes_) == 2:
    y_prob = svm_model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {roc_auc:.4f}")

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.show()

# Precision-Recall Curve
y_prob = svm_model.predict_proba(X_test)
precision = dict()
recall = dict()
for i in range(len(label_encoder.classes_)):
    precision[i], recall[i], _ = precision_recall_curve(y_test == i, y_prob[:, i])
    plt.plot(recall[i], precision[i], lw=2, label=f'Class {label_encoder.classes_[i]}')

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall curve")
plt.legend(loc="best")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from skopt import BayesSearchCV
from skopt.space import Real, Integer as SkoptInteger
from xgboost import XGBRegressor

def train_model(df):
    X = df[['International Reputation', 'Stamina','Strength','Aggression',
            'Composure','BallControl', 'Dribbling','Acceleration','Agility','Vision', 'LongPassing', 'Skill Moves', 
            'ShortPassing', 'ShotPower','Reactions']]
    y = df['Potential']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    param_space = {
        'n_estimators': SkoptInteger(100, 500),
        'learning_rate': Real(0.01, 0.3, prior='log-uniform'),
        'subsample': Real(0.6, 1.0),
        'max_depth': SkoptInteger(3, 10),
        'colsample_bytree': Real(0.5, 1.0)
    }

    bayes_search = BayesSearchCV(
        estimator=XGBRegressor(),
        search_spaces=param_space,
        n_iter=32,
        cv=5,
        scoring='r2',
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    bayes_search.fit(X_train, y_train)
    print("Best parameters found: ", bayes_search.best_params_)
    print("Best cross-validation R^2 score: ", bayes_search.best_score_)

    y_pred = bayes_search.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    accuracy = r2 * 100
    print('Test Set R-squared: ', r2)
    print('Test Set Mean Absolute Error: ', mae)
    print('Test Set Accuracy: {:.2f}%'.format(accuracy))

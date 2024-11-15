import pandas as pd
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report

# Class to preprocess the data
class DataPreprocessor:
    def __init__(self, filename):
        self.df = pd.read_csv(filename)

    # Handle missing values by dropping rows with missing values in the 'CreditScore' column.
    def handle_missing_values(self):
        self.df.dropna(subset=['CreditScore'], inplace=True)

    # Drop irrelevant columns such as 'Unnamed: 0', 'id', 'CustomerId', and 'Surname'
    def drop_unrelevant_columns(self):
        self.df.drop(columns=['Unnamed: 0', 'id', 'CustomerId', 'Surname'], inplace=True)
    
    # Split the data into training and testing sets
    def split_data(self):
        input = self.df.drop('churn', axis=1)
        output = self.df['churn']
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(input, output, test_size=0.2, random_state=42)

# Class to perform feature encoding
class FeatureEncoder:
    def __init__(self):
        self.BinaryEncode = {"Gender": {"Male": 1, "Female": 0}}
        self.train_encoded_geo = None

    # Perform binary encoding for the 'Gender' column
    def binary_encoding(self, df_train, df_test):
        df_train = df_train.replace(self.BinaryEncode)
        df_test = df_test.replace(self.BinaryEncode)
        return df_train, df_test

    #Perform one-hot encoding for the 'Geography' column
    def onehot_encoding(self, df_train, df_test):   
        geo_enc_train = df_train[['Geography']]
        geo_enc_test = df_test[['Geography']]
        self.train_encoded_geo = OneHotEncoder()
        geo_enc_train = pd.DataFrame(self.train_encoded_geo.fit_transform(geo_enc_train).toarray(), columns=self.train_encoded_geo.get_feature_names_out())
        geo_enc_test = pd.DataFrame(self.train_encoded_geo.transform(geo_enc_test).toarray(), columns=self.train_encoded_geo.get_feature_names_out())
        df_train = df_train.reset_index()
        df_test = df_test.reset_index()
        df_train_enc = pd.concat([df_train, geo_enc_train], axis=1)
        df_train_enc = df_train_enc.drop(['Geography'], axis=1)
        df_test_enc = pd.concat([df_test, geo_enc_test], axis=1)
        df_test_enc = df_test_enc.drop(['Geography'], axis=1)
        return df_train_enc, df_test_enc

    # Save the binary encoder to a pickle file
    def save_encoders(self, binary_filename):
        with open(binary_filename, 'wb') as file:
            pickle.dump(self.BinaryEncode, file)

# Class to perform feature scaling
class FeatureScaler:
    def __init__(self):
        self.ColScale = ['CreditScore', 'Age', 'Balance', 'EstimatedSalary']
        self.Scaler = StandardScaler()
    
    # Scale the specified features using standardization
    def scale_features(self, df_train, df_test):
        df_train[self.ColScale] = self.Scaler.fit_transform(df_train[self.ColScale])
        df_test[self.ColScale] = self.Scaler.transform(df_test[self.ColScale])
        return df_train, df_test

# Class to train the XGBoost model
class ModelTrainer:
    def __init__(self):
        self.param_grid_xgb = {'n_estimators': [50, 100],
                               'max_depth': [3, 5],
                               'learning_rate': [0.1, 0.01],
                               'subsample': [0.5, 0.7]
                               }
        self.xgb_classifier_tuned = XGBClassifier(random_state=42)

    # Tune the XGBoost model using grid search
    def tune_model(self, x_train, y_train):
        xgb_classifier_tuned = GridSearchCV(self.xgb_classifier_tuned,
                                            param_grid=self.param_grid_xgb,
                                            scoring='accuracy',
                                            cv=5)
        xgb_classifier_tuned.fit(x_train, y_train)
        return xgb_classifier_tuned.best_estimator_

# Class to evaluate the trained model
class ModelEvaluator:
    def evaluate_model(self, model, x_test, y_test):
        """Evaluate the model's performance on the test data."""
        y_predict = model.predict(x_test)
        print(classification_report(y_test, y_predict))

# Class to save the trained model
class ModelSaver:
    def save_model(self, model, filename):
        """Save the trained model to a pickle file."""
        with open(filename, 'wb') as file:
            pickle.dump(model, file)

def main():
    # Data Preprocessing
    data_processor = DataPreprocessor("data_C.csv")
    data_processor.handle_missing_values()
    data_processor.drop_unrelevant_columns()
    data_processor.split_data()

    # Feature Engineering
    feature_encoder = FeatureEncoder()
    x_train_enc, x_test_enc = feature_encoder.binary_encoding(data_processor.x_train, data_processor.x_test)
    x_train_enc, x_test_enc = feature_encoder.onehot_encoding(x_train_enc, x_test_enc)
    feature_encoder.save_encoders('gender_encoder.pkl')

    feature_scaler = FeatureScaler()
    x_train_scaled, x_test_scaled = feature_scaler.scale_features(x_train_enc, x_test_enc)

    # Modelling (XGBoost)
    model_trainer = ModelTrainer()
    xgb_model = model_trainer.tune_model(x_train_scaled, data_processor.y_train)

    # Model Evaluation
    model_evaluator = ModelEvaluator()
    model_evaluator.evaluate_model(xgb_model, x_test_scaled, data_processor.y_test)

    # Save Model
    model_saver = ModelSaver()
    model_saver.save_model(xgb_model, 'xgb_classifier_model.pkl')

if __name__ == "__main__":
    main()
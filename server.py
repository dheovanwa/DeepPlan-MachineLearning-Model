import flask
from flask import Flask, request, jsonify
import pickle
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd

app = Flask(__name__)

# Load the trained models
trained_models = {}
model_dir = 'Models'
for filename in os.listdir(model_dir):
    if filename.endswith(".pkl"):
        target_name = filename.replace('_model.pkl', '')
        with open(os.path.join(model_dir, filename), 'rb') as f:
            trained_models[target_name] = pickle.load(f)
        print(f"Loaded model for {target_name}")

# Define target variables (re-defined as they are needed for preprocessing setup)
target_variables = ['biaya_akhir_riil_miliar_rp', 'durasi_akhir_riil_hari', 'profit_margin_riil_persen', 'terjadi_pembengkakan_biaya_signifikan', 'terjadi_keterlambatan_signifikan']

# Recreate and fit the preprocessor
# In a real-world scenario, you would save and load the fitted preprocessor.
# For this example, we will recreate it using a sample of the data structure.
# Ideally, you would fit this on your full training data.
# Since we have df available from previous execution, we can use it to recreate preprocessor.
# If df was not available, you would need to load a sample or the full training data.

# Identify categorical and numerical columns (excluding targets and project_id)
# Assuming the original df structure is needed to define features for preprocessor
# If df is not available, you would need to define these based on expected input data structure
# For this fix, we assume df is available from the previous cell execution context.

df = pd.read_csv("dataset_proyek_konstruksi.csv")

categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
if 'project_id' in categorical_features:
    categorical_features.remove('project_id')

numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_features = [col for col in numerical_features if col not in target_variables and col != 'project_id']

# Create and fit the column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])

# Fit the preprocessor. In a production setting, you would fit this on your training data.
# Here, we fit it on the available df, dropping target variables and project_id
preprocessor.fit(df.drop(columns=target_variables + ['project_id']))


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        df_new = pd.DataFrame(data)

        # Ensure the columns in the new data match the training data columns (excluding targets and project_id)
        # and are in the same order
        # This assumes you have a list of the original feature columns available
        # Let's use the columns from the original X dataframe (excluding targets and project_id)
        original_features = [col for col in df.columns if col not in target_variables + ['project_id']]
        df_new = df_new[original_features]


        # Preprocess the new data
        X_new_processed = preprocessor.transform(df_new)

        predictions = {}
        for target, model in trained_models.items():
            prediction = model.predict(X_new_processed)
            # Convert predictions to list for JSON serialization
            predictions[target] = prediction.tolist()

        return jsonify(predictions)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Run the Flask app. In a Colab environment, you might need to use a tool like ngrok
    # to expose your local server to the internet for testing.
    # For local testing within Colab, you can use the default run.
    app.run(debug=True)
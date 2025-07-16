# Asegúrate de tener estas librerías instaladas en tu entorno de desarrollo
# pip install google-cloud-aiplatform kfp google-cloud-bigquery pandas scikit-learn imbalanced-learn matplotlib joblib db-dtypes google-auth

import kfp
from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model, Metrics, Artifact
from google.cloud import aiplatform
import os
import time
import json

# --- Configuración del Pipeline ---
PROJECT_ID = "data-analytics-454017"  # ¡Cambia esto por tu Project ID!
REGION = "us-central1"  # ¡Cambia esto por tu región!
# Corrected PIPELINE_ROOT to use PROJECT_ID and ensure it's a valid bucket name format
PIPELINE_ROOT = f"gs://{PROJECT_ID}-vertex-pipelines-root/predictive_maintenance"
PIPELINE_NAME = "predictive-maintenance-pipeline"
PIPELINE_JSON = f"{PIPELINE_NAME}.json"  # Corrected filename

# --- Service Account Configuration for Pipeline Submission ---
# For Method 2: Impersonation
# This is the SA that will be impersonated to SUBMIT the pipeline job.
# The Workbench SA needs "Service Account Token Creator" role on this SA.
TARGET_SA_EMAIL_FOR_SUBMISSION = "pipeline-submitter-sa@data-analytics-454017.iam.gserviceaccount.com"  # <--- !!! UPDATE THIS EMAIL !!!

# Service account for the pipeline EXECUTION (i.e., what the components run as)
# If None, Vertex AI default service agent will be used.
# The TARGET_SA_EMAIL_FOR_SUBMISSION needs 'Service Account User' role on this execution SA.
PIPELINE_EXECUTION_SA = (
    "pipeline-submitter-sa@data-analytics-454017.iam.gserviceaccount.com"  # or "your-pipeline-execution-sa@your-project-id.iam.gserviceaccount.com"
)

# Asegúrate de que el bucket para PIPELINE_ROOT existe
# Puedes crearlo con: gsutil mb -p {PROJECT_ID} gs://{PROJECT_ID}-vertex-pipelines-root/
# (Replace {PROJECT_ID} with your actual project ID in the gsutil command)

# --- Componente 1: Cargar y Preprocesar Datos de Entrenamiento ---
@dsl.component(
    base_image="us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-6:latest",
    packages_to_install=[
        "imbalanced-learn==0.13.0",
        "google-cloud-bigquery",
        "db-dtypes==1.1.1",
    ],
)
def preprocess_train_data_op(
    project_id: str,
    bq_source_uri_train: str,
    processed_x_train: Output[Dataset],
    processed_y_train: Output[Dataset],
    scaler_artifact: Output[Artifact],
    encoder_artifact: Output[
        Artifact
    ],  # Renamed from encoder_columns_artifact for consistency
):
    import pandas as pd
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from imblearn.over_sampling import SMOTE
    from google.cloud import bigquery
    import joblib

    client = bigquery.Client(project=project_id)
    # Corrected BQ query syntax using backticks for table names
    sql_train = f"SELECT * FROM `{bq_source_uri_train}`"
    df_train = client.query(sql_train).to_dataframe()

    print("Datos de entrenamiento cargados:")
    print(df_train.head())

    dtypes = {
        "id": "int64",
        "Product_ID": "object",
        "Type": "object",
        "Air_temperature": "float64",
        "Process_temperature": "float64",
        "Rotational_speed": "int64",
        "Torque": "float64",
        "Tool_wear": "int64",
        "Machine_failure": "int64",
        "TWF": "int64",
        "HDF": "int64",
        "PWF": "int64",
        "OSF": "int64",
        "RNF": "int64",
    }
    existing_cols = {
        col: dtype for col, dtype in dtypes.items() if col in df_train.columns
    }
    df_train = df_train.astype(existing_cols)

    print("Tipos de datos después de la conversión:")
    print(df_train.dtypes)

    X = df_train.drop(["id", "Machine_failure", "Product_ID"], axis=1)
    y = df_train["Machine_failure"]

    categorical_cols = ["Type"]
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first")

    X_encoded_np = encoder.fit_transform(X[categorical_cols])
    encoded_cols = encoder.get_feature_names_out(categorical_cols)
    X_encoded_df = pd.DataFrame(X_encoded_np, columns=encoded_cols, index=X.index)

    X = X.drop(columns=categorical_cols)
    X = pd.concat([X, X_encoded_df], axis=1)

    print("Columnas después de One-Hot Encoding:")
    print(X.columns)

    joblib.dump(encoder, encoder_artifact.path)  # Guardamos el encoder completo

    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X, y)

    if not isinstance(X_smote, pd.DataFrame):
        X_smote = pd.DataFrame(X_smote, columns=X.columns)

    scaler = StandardScaler()
    X_smote_scaled_np = scaler.fit_transform(X_smote)
    X_smote_scaled = pd.DataFrame(X_smote_scaled_np, columns=X_smote.columns)

    joblib.dump(scaler, scaler_artifact.path)

    X_smote_scaled.to_csv(processed_x_train.path, index=False)
    y_smote.to_csv(processed_y_train.path, index=False, header=True)

    print(f"X_train procesado guardado en: {processed_x_train.path}")
    print(f"y_train procesado guardado en: {processed_y_train.path}")
    print(f"Scaler guardado en: {scaler_artifact.path}")
    print(f"Encoder guardado en: {encoder_artifact.path}")


# --- Componente 2: Entrenar Modelo ---
@dsl.component(
    base_image="us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-6:latest",
)
def train_model_op(
    processed_x_train: Input[Dataset],
    processed_y_train: Input[Dataset],
    trained_model: Output[Model],
    training_metrics: Output[Metrics],
):
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    import joblib

    X_train = pd.read_csv(processed_x_train.path)
    y_train = pd.read_csv(processed_y_train.path).squeeze()

    model = LogisticRegression(random_state=42, class_weight="balanced", max_iter=1000)
    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)
    y_prob_train = model.predict_proba(X_train)[:, 1]
    auc_train = roc_auc_score(y_train, y_prob_train)

    print(f"Model training score: {train_score}")
    print(f"Model training AUC: {auc_train}")

    training_metrics.log_metric("train_accuracy", train_score)
    training_metrics.log_metric("train_auc", auc_train)

    joblib.dump(model, trained_model.path)
    print(f"Modelo entrenado guardado en: {trained_model.path}")


# --- Componente 3: Evaluar y Graficar ROC ---
@dsl.component(
    base_image="us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-6:latest",
    packages_to_install=[
        "matplotlib==3.7.2"
    ],
)
def evaluate_and_plot_roc_op(
    trained_model_input: Input[Model],
    processed_x_train: Input[Dataset],
    processed_y_train: Input[Dataset],
    roc_plot: Output[Artifact],
    evaluation_metrics: Output[Metrics],
):
    import pandas as pd
    from sklearn.metrics import roc_auc_score, roc_curve
    import matplotlib.pyplot as plt
    import joblib

    model = joblib.load(trained_model_input.path)
    X_train = pd.read_csv(processed_x_train.path)
    y_train = pd.read_csv(processed_y_train.path).squeeze()

    y_prob = model.predict_proba(X_train)[:, 1]
    auc = roc_auc_score(y_train, y_prob)
    fpr, tpr, _ = roc_curve(y_train, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", label=f"ROC curve (area = {auc:.2f})")
    plt.plot([0, 1], [0, 1], color="red", linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (Training Data)")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(roc_plot.path)
    plt.close()

    evaluation_metrics.log_metric("final_train_auc", auc)
    print(f"AUC-ROC (Training Data): {auc:.4f}")
    print(f"Gráfica ROC guardada en: {roc_plot.path}")


# --- Componente 4: Predecir en Datos de Prueba ---
@dsl.component(
    base_image="us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-6:latest",
    packages_to_install=[
        "google-cloud-bigquery",
        "db-dtypes",
    ],
)

def predict_on_test_data_op(
    project_id: str,
    bq_source_uri_test: str,
    trained_model: Input[Model],
    scaler_artifact: Input[Artifact],
    encoder_artifact: Input[Artifact],
    predictions_csv: Output[Dataset],
):
    import pandas as pd
    from google.cloud import bigquery
    import joblib

    client = bigquery.Client(project=project_id)
    # Corrected BQ query syntax
    sql_test = f"SELECT * FROM `{bq_source_uri_test}`"
    df_test_original = client.query(sql_test).to_dataframe()

    print("Datos de prueba cargados:")
    print(df_test_original.head())

    model = joblib.load(trained_model.path)
    scaler = joblib.load(scaler_artifact.path)
    encoder = joblib.load(encoder_artifact.path)

    dtypes_test = {
        "id": "int64",
        "Product_ID": "object",
        "Type": "object",
        "Air_temperature": "float64",
        "Process_temperature": "float64",
        "Rotational_speed": "int64",
        "Torque": "float64",
        "Tool_wear": "int64",
        "TWF": "int64",
        "HDF": "int64",
        "PWF": "int64",
        "OSF": "int64",
        "RNF": "int64",
    }
    existing_cols_test = {
        col: dtype
        for col, dtype in dtypes_test.items()
        if col in df_test_original.columns
    }
    df_test_processed = df_test_original.astype(existing_cols_test)

    # Prepare X_test, be careful if 'Product_ID' is not always present
    cols_to_drop = ["id"]
    if "Product_ID" in df_test_processed.columns:
        cols_to_drop.append("Product_ID")
    X_test = df_test_processed.drop(columns=cols_to_drop, axis=1, errors="ignore")

    categorical_cols = ["Type"]
    # Robust handling of 'Type' column for encoding
    if "Type" in X_test.columns:
        X_test_encoded_np = encoder.transform(X_test[categorical_cols])
        encoded_cols = encoder.get_feature_names_out(categorical_cols)
        X_test_encoded_df = pd.DataFrame(
            X_test_encoded_np, columns=encoded_cols, index=X_test.index
        )

        X_test = X_test.drop(columns=categorical_cols)
        X_test = pd.concat([X_test, X_test_encoded_df], axis=1)
    else:
        # If 'Type' was in training but not test, add empty encoded columns
        # This assumes encoder was fit on data that included 'Type'
        print(
            f"Warning: Categorical column '{categorical_cols[0]}' not found in test data. Adding empty encoded columns."
        )
        encoded_cols_from_encoder = encoder.get_feature_names_out(categorical_cols)
        for col_name in encoded_cols_from_encoder:
            if col_name not in X_test.columns:
                X_test[col_name] = 0

    # Align columns with training data (after encoding, before scaling)
    train_cols_from_scaler = (
        scaler.feature_names_in_
    )  # These are the columns the scaler was fit on

    missing_cols = set(train_cols_from_scaler) - set(X_test.columns)
    for c in missing_cols:
        print(
            f"Warning: Column '{c}' was in training data but not in test data after encoding. Adding it with 0s."
        )
        X_test[c] = 0

    # Ensure correct column order and remove any extra columns not in training
    X_test = X_test[train_cols_from_scaler]

    X_test_scaled_np = scaler.transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_scaled_np, columns=X_test.columns)

    y_test_prob = model.predict_proba(X_test_scaled)[:, 1]

    final_result = pd.DataFrame(
        {"id": df_test_original["id"], "Machine_failure": y_test_prob}
    )

    final_result.to_csv(predictions_csv.path, index=False)
    print(f"Predicciones guardadas en: {predictions_csv.path}")
    print(final_result.head())


# --- Definición del Pipeline ---
@dsl.pipeline(
    name=PIPELINE_NAME,
    description="Pipeline para predecir fallos de máquina.",
    pipeline_root=PIPELINE_ROOT,
)
def predictive_maintenance_pipeline(
    project_id: str = PROJECT_ID,
    bq_source_train: str = "data-analytics-454017.modelos.train",
    bq_source_test: str = "data-analytics-454017.modelos.test",
):
    preprocess_task = preprocess_train_data_op(
        project_id=project_id, bq_source_uri_train=bq_source_train
    )

    train_task = train_model_op(
        processed_x_train=preprocess_task.outputs["processed_x_train"],
        processed_y_train=preprocess_task.outputs["processed_y_train"],
    )

    evaluate_task = evaluate_and_plot_roc_op(
        trained_model_input=train_task.outputs["model.joblib"],
        processed_x_train=preprocess_task.outputs["processed_x_train"],
        processed_y_train=preprocess_task.outputs["processed_y_train"],
    )

    predict_task = predict_on_test_data_op(
        project_id=project_id,
        bq_source_uri_test=bq_source_test,
        trained_model=train_task.outputs["model.joblib"],
        scaler_artifact=preprocess_task.outputs["scaler_artifact"],
        encoder_artifact=preprocess_task.outputs[
            "encoder_artifact"
        ],  # Corrected artifact name
    )
    # evaluate_task.after(train_task) # This is usually implicit due to data dependency


# --- Compilar y Ejecutar el Pipeline ---
if __name__ == "__main__":
    # Import necessary auth libraries for impersonation
    import google.auth
    import google.auth.impersonated_credentials

    # Compile the pipeline
    kfp.compiler.Compiler().compile(
        pipeline_func=predictive_maintenance_pipeline, package_path=PIPELINE_JSON
    )
    print(f"Pipeline compilado en {PIPELINE_JSON}")

    # --- METHOD 2: Using Service Account Impersonation ---
    print(
        f"Attempting to submit pipeline by impersonating SA: {TARGET_SA_EMAIL_FOR_SUBMISSION}"
    )

    try:
        # 1. Get default credentials (of the Workbench SA or the environment it's running in)
        #    These credentials need the "Service Account Token Creator" role on TARGET_SA_EMAIL_FOR_SUBMISSION.
        print("Fetching default credentials for impersonation source...")
        default_creds, _ = google.auth.default(
            scopes=[
                "https://www.googleapis.com/auth/cloud-platform"
            ]  # Broad scope for impersonation
        )

        # 2. Create impersonated credentials for the target service account
        print(
            f"Creating impersonated credentials for target SA: {TARGET_SA_EMAIL_FOR_SUBMISSION}..."
        )
        impersonated_creds = google.auth.impersonated_credentials.Credentials(
            source_credentials=default_creds,
            target_principal=TARGET_SA_EMAIL_FOR_SUBMISSION,
            target_scopes=[
                "https://www.googleapis.com/auth/cloud-platform"
            ],  # Scopes for the impersonated SA
            # lifetime=3600 # Optional: token lifetime in seconds (default is 1 hour)
        )

        # 3. Initialize the AI Platform client WITH THE IMPERSONATED CREDENTIALS
        #    All subsequent aiplatform API calls will use these impersonated credentials.
        print("Initializing AI Platform client with impersonated credentials...")
        aiplatform.init(
            project=PROJECT_ID, location=REGION, credentials=impersonated_creds
        )

        # 4. Define the PipelineJob
        #    The display_name is how it appears in the Vertex AI UI.
        #    The service_account here is for the *execution* of pipeline steps.
        job_display_name = (
            f"{PIPELINE_NAME.replace('_', '-')}-{time.strftime('%Y%m%d-%H%M%S')}"
        )
        job = aiplatform.PipelineJob(
            display_name=job_display_name,
            template_path=PIPELINE_JSON,
            pipeline_root=PIPELINE_ROOT,
            enable_caching=True,
            # parameter_values={...} # If you need to override pipeline parameters
        )

        print(
            f"Enviando el pipeline job '{job_display_name}' a Vertex AI con el Service Account impersonado..."
        )

        # 5. Submit the job.
        #    The `service_account` parameter here specifies the SA that the *pipeline components will run as*.
        #    The API call to submit the job itself is authenticated by `impersonated_creds`.
        job.submit(service_account=PIPELINE_EXECUTION_SA)

        print(f"Pipeline job enviado exitosamente!")
        print(f"Puedes verlo en la consola de Vertex AI: {job._dashboard_uri()}")

    except google.auth.exceptions.RefreshError as re:
        print(f"Error de autenticación al intentar impersonar: {re}")
        print(
            "Verifica que el Service Account del Notebook tenga el rol 'Service Account Token Creator' sobre el SA: "
            + TARGET_SA_EMAIL_FOR_SUBMISSION
        )
    except Exception as e:
        print(f"Error submitting pipeline via impersonation: {e}")
        print("Detalles del error:", type(e), e.args)


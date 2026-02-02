import pandas as pd                     
import streamlit as st                  # For building interactive web apps
from sqlalchemy import create_engine    # For database connection
from urllib.parse import quote
import pickle, joblib                   # For loading model and pipeline
from PIL import Image                   # For displaying logo/image


# ============================================================
# LOAD TRAINED MODEL & PREPROCESSING PIPELINE
# ============================================================

model = pickle.load(open('knn_glass_model.pkl', 'rb'))
preprocessing_pipeline = joblib.load(
    'pipeline_with_feature_selection_glass.pkl'
)  # Includes preprocessing + feature selection


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict(data, user, pw, db):
    try:
        # Create MySQL engine
        engine = create_engine(
            f"mysql+pymysql://{user}:{quote(pw)}@localhost/{db}"
        )

        # Preprocess input data
        processed_data = preprocessing_pipeline.transform(data)

        # Predict glass type
        predictions = pd.DataFrame(
            model.predict(processed_data),
            columns=['Predicted_Glass_Type']
        )

        # Combine prediction with input data
        final = pd.concat(
            [predictions, data.reset_index(drop=True)],
            axis=1
        )

        # Save predictions to MySQL
        try:
            final.to_sql(
                'glass_predictions',
                con=engine,
                if_exists='replace',
                chunksize=1000,
                index=False
            )
            st.success("Predictions successfully saved to MySQL database ✅")

        except Exception as db_error:
            st.error(f"❌ Failed to save predictions to database: {db_error}")

        return final

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        return pd.DataFrame()


# ============================================================
# STREAMLIT APP INTERFACE
# ============================================================

def main():

    # Sidebar logo
    image = Image.open("AiSPRY logo.jpg")
    st.sidebar.image(image)

    # Titles
    st.title("Glass Type Classification")
    st.sidebar.title("Glass Type Prediction using KNN")

    # File upload
    uploadedFile = st.sidebar.file_uploader(
        "Upload Glass Data (CSV or Excel)",
        type=['csv', 'xlsx'],
        accept_multiple_files=False
    )

    if uploadedFile is not None:
        try:
            data = pd.read_csv(uploadedFile)
        except:
            try:
                data = pd.read_excel(uploadedFile)
            except:
                data = pd.DataFrame(uploadedFile)
    else:
        st.sidebar.warning("Please upload a CSV or Excel file.")

    # MySQL credentials
    user = st.sidebar.text_input("MySQL User", "Type Here")
    pw = st.sidebar.text_input("MySQL Password", "Type Here", type='password')
    db = st.sidebar.text_input("Database Name", "Type Here")

    # Predict button
    if st.button("Predict Glass Type"):
        result = predict(data, user, pw, db)

        if not result.empty:
            import seaborn as sns
            cm = sns.light_palette("green", as_cmap=True)
            st.table(result.style.background_gradient(cmap=cm))


# ============================================================
# APP ENTRY POINT
# ============================================================

if __name__ == '__main__':
    main()



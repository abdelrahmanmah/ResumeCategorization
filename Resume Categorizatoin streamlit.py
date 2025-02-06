import streamlit as st
import numpy as np
import pickle
import docx
import PyPDF2
import re
import base64  # Import base64 for encoding the notebook file
import pandas as pd


# Load pre-trained model and TF-IDF vectorizer
knn_model = pickle.load(open('E:\\ERU\\Level 4\\S1\\ML\\Project\\.venv\\knn.pkl', 'rb'))
svc_model = pickle.load(open('E:\\ERU\\Level 4\\S1\\ML\\Project\\.venv\\svc.pkl', 'rb'))
rf_model = pickle.load(open('E:\\ERU\\Level 4\\S1\\ML\\Project\\.venv\\rf.pkl', 'rb'))
model = pickle.load(open('E:\\ERU\\Level 4\\S1\\ML\\Project\\.venv\\mlp.pkl', 'rb'))
ensemble_model = pickle.load(open('E:\\ERU\\Level 4\\S1\\ML\\Project\\.venv\\ensemble.pkl', 'rb'))
tfidf = pickle.load(open('E:\\ERU\\Level 4\\S1\\ML\\Project\\.venv\\tfidf.pkl', 'rb'))
le = pickle.load(open('E:\\ERU\\Level 4\\S1\\ML\\Project\\.venv\\encoder.pkl', 'rb'))

# Function to clean resume text (same as before)
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

# Functions for extracting text (same as before)
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text

def extract_text_from_txt(file):
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        text = file.read().decode('latin-1')
    return text

# Function to handle file upload and extraction (same as before)
def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        text = extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        text = extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        text = extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")
    return text

# Function to predict the category of a resume and print results for each model
def pred(input_resume):
    # Preprocess the input text (e.g., cleaning, etc.)
    cleaned_text = cleanResume(input_resume)

    # Vectorize the cleaned text using the same TF-IDF vectorizer used during training
    vectorized_text = tfidf.transform([cleaned_text])

    # Convert sparse matrix to dense
    vectorized_text = vectorized_text.toarray()

    # Prediction
    predicted_category_knn = knn_model.predict(vectorized_text)
    predicted_category_svc = svc_model.predict(vectorized_text)
    predicted_category_rf = rf_model.predict(vectorized_text)
    predicted_category_mlp = np.argmax(model.predict(vectorized_text), axis=1)
    predicted_category_ensemble = ensemble_model.predict(vectorized_text)

    # Get name of predicted category for each model
    category_knn = le.inverse_transform(predicted_category_knn)[0]
    category_svc = le.inverse_transform(predicted_category_svc)[0]
    category_rf = le.inverse_transform(predicted_category_rf)[0]
    category_mlp = le.inverse_transform(predicted_category_mlp)[0]
    category_ensemble = le.inverse_transform(predicted_category_ensemble)[0]

    # Return all predictions as a dictionary
    return {
        "KNN": category_knn,
        "SVC": category_svc,
        "Random Forest": category_rf,
        "MLP": category_mlp,
        "Ensemble": category_ensemble
    }

# Function to display the main app page
def main_app_page():
    st.title("Resume Categorization App 🔍🔮")
    st.markdown("Upload a resume in PDF, TXT, or DOCX format and get the predicted job category.")

    uploaded_file = st.file_uploader("Upload a Resume", type=["pdf", "docx", "txt"])

    if uploaded_file is not None:
        try:
            resume_text = handle_file_upload(uploaded_file)
            st.write("Successfully extracted the text from the uploaded resume.")
            if st.checkbox("Show extracted text", False):
                st.text_area("Extracted Resume Text", resume_text, height=300)

            st.subheader("Predicted Categories by Each Model")
            predictions = pred(resume_text)

            # Create a DataFrame for the results
            results_data = {
                "Model Type": ["Machine Learning", "Machine Learning", "Machine Learning", "Deep Learning", "Ensemble"],
                "Model": ["KNN", "SVC", "Random Forest", "MLP", "Ensemble"],
                "Predicted Category": [predictions["KNN"], predictions["SVC"], predictions["Random Forest"],
                                         predictions["MLP"], predictions["Ensemble"]]
            }
            results_df = pd.DataFrame(results_data)

            # Display the results in a table
            st.table(results_df)


        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")


# Function to display the intro page
def intro_page():
    st.title("Welcome to the Resume Categorization Project! 👋")
    st.header("Our Team ❤️‍🔥")

    team_members = [
        {"name": "Abdelrahman Mahmoud", "portfolio": "https://abdelrahmanmah.github.io/SafeZoneInc/Abdelrahman.html"},
        {"name": "GannaTullah Gouda", "portfolio": "https://gannaasaad.github.io/index.html"},
        {"name": "Ahmed Khaled", "portfolio": "https://holako22.github.io/"},
        {"name": "Ali Mohamed", "portfolio": "https://aliiimohamedaliii.github.io/My-portfolio/"},
        # Add other team members
    ]

    for member in team_members:
        st.markdown(f"- **{member['name']}**: [Portfolio]({member['portfolio']})")

    st.markdown("---")

    st.subheader("About this Project 📚")
    st.markdown("""
This project tackles the challenge of **resume categorization** using machine learning and deep learning techniques. Companies often face the daunting task of sifting through numerous resumes for each job opening. This app aims to automate and streamline this process by predicting the job category a given resume belongs to. By using a trained model, the app can quickly suggest the appropriate job category for each resume, saving time and resources for recruiters.

The project follows these key steps:

1.  **Dataset Acquisition:** The project uses the [Resume Dataset](https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset) from Kaggle. This dataset consists of resumes categorized into 25 distinct job fields, providing a solid foundation for training our models.

2.  **Text Preprocessing:** Raw resume text is often messy. We apply preprocessing techniques to clean the resume data, including:
    *   Removing URLs, RTs, hashtags, and mentions
    *   Eliminating special characters and non-ASCII characters
    *   Collapsing extra whitespace

3.  **Data Preparation:**
    *   **Category Encoding:** We transform the textual categories into numerical labels using Label Encoding, allowing our machine learning algorithms to work with the data.
    *   **TF-IDF Vectorization:** We convert the cleaned text data into numerical vectors using TF-IDF (Term Frequency-Inverse Document Frequency), which gives more weight to words that are specific to a document in the dataset.

4. **Data Splitting:** The dataset is divided into training and testing sets to properly evaluate the performance of our models.

5.  **Model Development:** We build and train multiple models:
    *   **Machine Learning Models:**
        *   **K-Nearest Neighbors (KNN):** A simple yet effective classification algorithm based on distance between points.
        *   **Support Vector Machine (SVC):** A powerful algorithm that finds an optimal hyperplane to separate classes.
        *   **Random Forest:** An ensemble method that combines multiple decision trees for more robust predictions.
    *   **Deep Learning Model:**
        *   **Multilayer Perceptron (MLP):** A neural network model with multiple hidden layers to learn complex patterns from the data.
    *   **Ensemble Model**
         *   **Voting Classifier:** Combines the predictions from the machine learning models to achieve more accurate and robust results.

6.  **Model Evaluation:** Each model is evaluated using metrics like accuracy, precision, recall, and F1-score. We also visualize performance using confusion matrices and classification reports to understand where our models are performing well and where they could improve.

7. **Model Deployment:** The trained models, along with the TF-IDF vectorizer and label encoder, are serialized (pickled) for use in the Streamlit application. This step makes the models reusable without retraining each time, **and it is where Streamlit comes into play**. This app was built using Streamlit, a python library to build interactive and sharable web applications. Streamlit made it easier to build this application by handling the user interface, input/output and allowing to display the predictions of different models to be accessible through a web browser.

8.  **Prediction:** The app provides predictions from all the trained models, allowing for an extensive comparison of their results. The user receives a result from each model, including KNN, SVC, Random Forest, MLP, and the ensemble model.

Through these steps, this project demonstrates the application of machine learning and deep learning to solve a real-world problem of resume categorization and aims to provide an efficient tool for recruiters to streamline their selection process.

You can use the tabs above to upload your resume and explore our results.
""")


# Function to display the project notebook
def notebook_page():
    st.title("Project Notebook 📄")
    st.markdown("Here's the link to our project notebook:")
    notebook_link = "https://drive.google.com/file/d/1jaLllonQuM_3hsPfmCVAlf6lI5MVY8qX/view?usp=sharing" # Replace with link

    st.markdown(f"[Project Notebook]({notebook_link})")

# Main app setup with tabs
def main():
    st.set_page_config(page_title="Resume Category Prediction 🔍🔮", page_icon="🔍", layout="wide")
    st.sidebar.title("Navigation")

    pages = {
      "Introduction 🫡": intro_page,
      "Project Notebook 📄": notebook_page,
      "Resume Classifier 🔮🪄": main_app_page,
    }

    selected_page = st.sidebar.radio("Go to:", list(pages.keys()))
    pages[selected_page]()

if __name__ == "__main__":
    main()
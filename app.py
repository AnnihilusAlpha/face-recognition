import streamlit as st
import face_recognition
import os
import cv2
import numpy as np
import sqlite3
from io import BytesIO


# Set the page configuration (title, icon, etc.)
st.set_page_config(
    page_title="Missing Person Finder",  # Title of the webpage
    page_icon="🕵️‍♂️",  # Icon to display in the browser tab (optional)
    layout="wide",  # Layout options: "centered" or "wide"
)

# Now you can continue with the rest of your app
st.title("Missing Person Website")


# Function to handle image resizing (same as in your example)
def read_img(uploaded_file_or_path):
    # If a file-like object (uploaded via Streamlit), process it
    if hasattr(uploaded_file_or_path, 'read'):
        img_bytes = uploaded_file_or_path.read()  # Read the uploaded file
        img_array = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    else:
        # If it's a file path, use cv2.imread
        img = cv2.imread(uploaded_file_or_path)
    
    if img is None:
        raise ValueError("Failed to decode image")
    
    (h, w) = img.shape[:2]
    width = 500
    ratio = width / float(w)
    height = int(h * ratio)
    return cv2.resize(img, (width, height))

#creating known folder
def ensure_known_folder_exists():
    if not os.path.exists("known"):
        os.makedirs("known")

# Database setup
def connect_db():
    return sqlite3.connect('missing_persons.db')

# Create tables if not exist
def create_tables():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS missing_persons (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        height INTEGER,
        weight INTEGER,
        features TEXT,
        img_encoding BLOB
    )
    """)
    conn.commit()
    conn.close()

# Function to get all missing persons data from the database
def get_all_missing_persons():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT name, features, img_encoding FROM missing_persons")
    rows = cursor.fetchall()
    conn.close()
    
    missing_persons = []
    for row in rows:
        name = row[0]
        features = row[1]
        img_encoding = np.frombuffer(row[2], dtype=np.float64)
        missing_persons.append((name, features, img_encoding))
    
    return missing_persons

# Function to display all missing persons
def display_all_missing_persons():
    st.subheader("All Missing Persons")
    
    missing_persons = get_all_missing_persons()
    
    if not missing_persons:
        st.write("No missing persons recorded yet.")
        return
    
    for person in missing_persons:
        name, features, img_encoding = person
        # Fetch the image encoding, decode it to an image
        img = face_recognition.load_image_file(f"known/{name}.jpg")  # Assuming images are stored in the 'known' directory

        st.image(img, caption=f"{name}", width=150)
        st.write(f"**Features**: {features}")
        st.write("---")  # Divider between entries

# Function to store missing person data, for images with multiple faces
def store_missing_person(name, height, weight, features, uploaded_file):
    conn = sqlite3.connect('missing_persons.db')
    cursor = conn.cursor()

    # Ensure the directory exists
    if not os.path.exists("known"):
        os.makedirs("known")
    
    # Define the path to save the uploaded image
    img_path = os.path.join("known", f"{name}.jpg")
    
    # Save the uploaded image to the 'known' directory
    with open(img_path, "wb") as img_file:
        img_file.write(uploaded_file.getbuffer())
    
    # Process the image for encoding
    img = read_img(img_path)  # Now passing the correct image path to read_img()
    img_enc = face_recognition.face_encodings(img)[0]
    img_enc_blob = sqlite3.Binary(img_enc.tobytes())

    cursor.execute("""
    INSERT INTO missing_persons (name, height, weight, features, img_encoding)
    VALUES (?, ?, ?, ?, ?)
    """, (name, height, weight, features, img_enc_blob))

    conn.commit()
    conn.close()
    print(f"Missing person {name} stored successfully.")
    
    # Clear the form and refresh the page after submission
    st.success(f"Missing person {name} has been reported successfully!")
    st.rerun()  # Refresh the page to reset the form

# Function to search for a missing person
def search_missing_person(uploaded_file_or_path):
    conn = connect_db()
    cursor = conn.cursor()

    # Pass the uploaded file object or image path to read_img
    img = read_img(uploaded_file_or_path)
    unknown_face_encodings = face_recognition.face_encodings(img)

    if not unknown_face_encodings:
        st.error("No faces detected in the image!")
        return

    cursor.execute("SELECT id, name, img_encoding FROM missing_persons")
    rows = cursor.fetchall()

    matched_names = []

    for unknown_face_encoding in unknown_face_encodings:
        for row in rows:
            stored_name = row[1]
            stored_encoding = np.frombuffer(row[2], dtype=np.float64)
            results = face_recognition.compare_faces([stored_encoding], unknown_face_encoding)

            if results[0]:
                matched_names.append(stored_name)

    if matched_names:
        st.success(f"Match found: {', '.join(set(matched_names))}")
    else:
        st.error("No matches found.")
    
    conn.close()

# Streamlit UI
def app():
    st.title("A facial recognition-based platform to report and search for missing persons easily and effectively.")
    ensure_known_folder_exists()

    menu = ["Report Missing Person", "Search for Missing Person", "All Missing Persons"]
    choice = st.sidebar.selectbox("Select an option", menu)

    if choice == "Report Missing Person":
        st.subheader("Report Missing Person")
        name = st.text_input("Name of the Missing Person")
        height = st.number_input("Height (cm)")
        weight = st.number_input("Weight (kg)")
        features = st.text_area("Distinct Features/Marks")
        img = st.file_uploader("Upload Image", type=["jpg", "png"])

        if img:
            if st.button("Submit"):
                store_missing_person(name, height, weight, features, img)
                st.success(f"Missing person {name} reported successfully.")

    elif choice == "Search for Missing Person":
        st.subheader("Search for Missing Person")
        img = st.file_uploader("Upload Image", type=["jpg", "png"])

        if img:
            if st.button("Search"):
                search_missing_person(img)

    elif choice == "All Missing Persons":
        display_all_missing_persons()  # Call the function to display all missing persons

if __name__ == "__main__":
    create_tables()
    app()


#pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

#import torch

#print("Number of GPUs available: ", torch.cuda.device_count())
#print("GPU Name: ", torch.cuda.get_device_name())
import os
import pickle
import cv2
import pandas as pd
from datetime import datetime
import numpy as np
from deepface import DeepFace

# Constants
DATA_PATH = "data"
ENCODINGS_FILE = os.path.join(DATA_PATH, "encodings.pickle")
ATTENDANCE_FILE = "attendance.csv"
MODEL_NAME = "VGG-Face" # Options: VGG-Face, Facenet, OpenFace, DeepFace, DeepID, ArcFace, Dlib

# Ensure data directory exists
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

# Caching variables
_cache_encodings = None
_cache_mtime = 0

def load_encodings():
    """Load face encodings from pickle file with caching."""
    global _cache_encodings, _cache_mtime
    
    if not os.path.exists(ENCODINGS_FILE):
        return {"encodings": [], "names": [], "ids": []}
    
    # Check file modification time
    current_mtime = os.path.getmtime(ENCODINGS_FILE)
    
    # Reload only if file changed or cache is empty
    if _cache_encodings is None or current_mtime > _cache_mtime:
        try:
            with open(ENCODINGS_FILE, "rb") as f:
                _cache_encodings = pickle.load(f)
            _cache_mtime = current_mtime
            
            # Convert list of encodings to numpy array for vectorization if not already
            if isinstance(_cache_encodings["encodings"], list):
                 _cache_encodings["encodings"] = np.array(_cache_encodings["encodings"])
                 
            print("DEBUG: Encodings reloaded from disk.")
        except Exception as e:
            print(f"Error loading encodings: {e}")
            # Return empty structure on error to prevent crash
            return {"encodings": [], "names": [], "ids": []}
            
    return _cache_encodings

def save_encoding(name, student_id, image):
    """
    Generate and save face encoding for a new student using DeepFace.
    Returns True if successful, False if no face found or error.
    """
    try:
        # DeepFace expects path or numpy array
        # enforce_detection=True ensures we only register if a face is clearly visible
        embedding_objs = DeepFace.represent(img_path=image, model_name=MODEL_NAME, enforce_detection=True)
        
        if not embedding_objs:
            return False
            
        # Take the first face found
        encoding = embedding_objs[0]["embedding"]
        
        # Load existing data
        data = load_encodings()
        
        # We need to handle the conversion back and forth if we use numpy for cache
        current_encodings = data["encodings"]
        if isinstance(current_encodings, np.ndarray):
            current_encodings = current_encodings.tolist()
            
        # Append new data
        current_encodings.append(encoding)
        data["names"].append(name)
        data["ids"].append(student_id)
        
        # Save back to pickle with just lists (easier serializability)
        save_data = {
            "encodings": current_encodings,
            "names": data["names"],
            "ids": data["ids"]
        }
        
        with open(ENCODINGS_FILE, "wb") as f:
            pickle.dump(save_data, f)
        
        # Save the face image for reference
        img_path = os.path.join(DATA_PATH, f"{student_id}_{name}.jpg")
        cv2.imwrite(img_path, image)
        
        # Force cache refresh
        global _cache_encodings
        _cache_encodings = None
        
        return True
    except ValueError:
        # Face could not be detected
        return False
    except Exception as e:
        print(f"Error in save_encoding: {e}")
        return False

def mark_attendance_log(name, student_id):
    """Mark attendance in the CSV file."""
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    if not os.path.exists(ATTENDANCE_FILE):
        df = pd.DataFrame(columns=["ID", "Name", "Date", "Time"])
    else:
        df = pd.read_csv(ATTENDANCE_FILE)
    
    # Check if marked today
    already_marked = df[(df["ID"] == str(student_id)) & (df["Date"] == date_str)]
    
    if already_marked.empty:
        new_entry = pd.DataFrame([{"ID": student_id, "Name": name, "Date": date_str, "Time": time_str}])
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv(ATTENDANCE_FILE, index=False)
        return True, f"Attendance marked for {name} ({student_id}) at {time_str}"
    else:
        return False, f"{name} is already marked present today."

def get_attendance_history():
    """Return attendance history as a DataFrame."""
    if not os.path.exists(ATTENDANCE_FILE):
        return pd.DataFrame(columns=["ID", "Name", "Date", "Time"])
    df = pd.read_csv(ATTENDANCE_FILE)
    # Ensure ID is string for consistency
    df["ID"] = df["ID"].astype(str)
    return df

def recognize_face(face_image_roi):
    """
    Identify a face in the ROI against known encodings using Vectorized Cosine Similarity.
    Faster than looping.
    """
    data = load_encodings()
    known_encodings = data["encodings"]
    known_names = data["names"]
    known_ids = data["ids"]

    if len(known_encodings) == 0:
        return None, "No registered users"

    try:
        # 1. Get embedding for the input image (ROI)
        # detector_backend='skip' assumes we already cropped the face
        # enforce_detection=False prevents errors if DeepFace is picky about the crop
        embedding_objs = DeepFace.represent(
            img_path=face_image_roi,
            model_name=MODEL_NAME,
            enforce_detection=False,
            detector_backend="skip" 
        )
        
        if not embedding_objs:
             return None, "No face embedding generated"

        target_encoding = np.array(embedding_objs[0]["embedding"])
        
        # 2. Vectorized Cosine Distance Calculation
        # Cosine Distance = 1 - Cosine Similarity
        # Similarity = (A . B) / (||A|| * ||B||)
        
        # Ensure known_encodings is a numpy array
        if not isinstance(known_encodings, np.ndarray):
            known_encodings = np.array(known_encodings)
            
        # Norms
        target_norm = np.linalg.norm(target_encoding)
        known_norms = np.linalg.norm(known_encodings, axis=1)
        
        # Dot product
        dot_products = np.dot(known_encodings, target_encoding)
        
        # Cosine Similarities
        similarities = dot_products / (known_norms * target_norm)
        
        # Distances
        distances = 1 - similarities
        
        # Find best match
        min_dist_idx = np.argmin(distances)
        min_dist = distances[min_dist_idx]
        
        threshold = 0.40 # Threshold for VGG-Face
        
        if min_dist < threshold:
             return known_names[min_dist_idx], known_ids[min_dist_idx]
        else:
             return None, "Unknown"
            
    except Exception as e:
        print(f"Recognition error: {e}")
        return None, "Error"

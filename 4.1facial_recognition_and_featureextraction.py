import os
import cv2
import dlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


path = "scored_hon"

def detect_faces(img, face_detector):
    """
    Detects faces in the specified image using the specified face detector.
    Returns a list of detected faces.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray, 1)
    if len(faces) == 0:
        print("No faces detected.")
    else:
        print(f"{len(faces)} face(s) detected.")
    return faces, gray



def get_biggest_face(faces):
    """
    Finds the biggest face in the specified list of detected faces.
    Returns the biggest face or None if no faces were detected.
    """
    biggest_face = None
    max_area = 0
    for face in faces:
        face_area = (face.right() - face.left()) * (face.bottom() - face.top())
        if face_area > max_area:
            biggest_face = face
            max_area = face_area
    return biggest_face


def calculate_scores(gray, biggest_face, landmarks):
    '''
    calculates a symmetric score, lip fullness score, estimates the BMI of the person in the picture,
    and calculates the size of the eyes in relation to the face
    '''

    # Calculate the symmetry score and landmarks

    left_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
    right_points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
    left_mean_x = sum([point[0] for point in left_points]) / len(left_points)
    left_mean_y = sum([point[1] for point in left_points]) / len(left_points)
    right_mean_x = sum([point[0] for point in right_points]) / len(right_points)
    right_mean_y = sum([point[1] for point in right_points]) / len(right_points)
    symmetry_score = abs(left_mean_x - right_mean_x) / (biggest_face.right() - biggest_face.left())

    # Get the midline of the face
    left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
    right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]
    left_eye_center = np.mean(left_eye, axis=0).astype(int)
    right_eye_center = np.mean(right_eye, axis=0).astype(int)

    midline = [left_eye_center[0], left_eye_center[1], right_eye_center[0], right_eye_center[1]]

    # Calculate the lip fullness score
    upper_lip_center = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(50, 53)]
    lower_lip_center = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(58, 61)]
    upper_lip_distance = np.linalg.norm(np.array(upper_lip_center[0]) - np.array(upper_lip_center[2]))
    lower_lip_distance = np.linalg.norm(np.array(lower_lip_center[0]) - np.array(lower_lip_center[2]))
    lip_fullness_score = lower_lip_distance / upper_lip_distance

    # Calculate the size of the eyes in relation to the face
    eye_distance = np.linalg.norm(np.array(left_eye_center) - np.array(right_eye_center))
    face_width = biggest_face.right() - biggest_face.left()
    eye_size_score = eye_distance / face_width

    # Calculate the adiposity score
    jaw_width = landmarks.part(16).x - landmarks.part(0).x
    face_height = landmarks.part(8).y - landmarks.part(27).y
    adiposity_score = jaw_width / face_height

    # Calculate the Facial width-to-height ratio (FWHR)
    face_height = landmarks.part(8).y - landmarks.part(27).y
    face_width = biggest_face.right() - biggest_face.left()
    fwhr_score = face_width / face_height

    # Calculate the Interocular distance-to-eyeball height ratio
    eye_height = (landmarks.part(41).y + landmarks.part(40).y + landmarks.part(37).y + landmarks.part(38).y) / 4 - landmarks.part(27).y
    interocular_ratio = eye_distance / eye_height

    #nose chin lip ratio
    nose_point = (landmarks.part(30).x, landmarks.part(30).y)
    chin_point = (landmarks.part(8).x, landmarks.part(8).y)
    lip_point = (landmarks.part(62).x, landmarks.part(62).y)
    nose_chin_distance = np.linalg.norm(np.array(nose_point) - np.array(chin_point))
    nose_lip_distance = np.linalg.norm(np.array(nose_point) - np.array(lip_point))
    nose_chin_lip_ratio = nose_chin_distance / nose_lip_distance

    # Calculate the Facial thirds ratio
    hairline_to_eyebrows = landmarks.part(19).y - biggest_face.top()
    eyebrows_to_base_of_nose = landmarks.part(27).y - landmarks.part(19).y
    base_of_nose_to_bottom_of_chin = biggest_face.bottom() - landmarks.part(8).y
    facial_thirds_ratio = [hairline_to_eyebrows / (biggest_face.bottom() - biggest_face.top()),
                           eyebrows_to_base_of_nose / (biggest_face.bottom() - biggest_face.top()),
                           base_of_nose_to_bottom_of_chin / (biggest_face.bottom() - biggest_face.top())]

    # Return the symmetry, lip fullness, adiposity scores, midline, and eye size score
    return symmetry_score, lip_fullness_score, adiposity_score, eye_size_score, eye_distance, midline, fwhr_score, \
           interocular_ratio, nose_chin_lip_ratio, facial_thirds_ratio




def detect_full_body(gray):
    # Load the Haar cascade classifier for full body detection
    classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_fullbody.xml")

    # Detect the full body in the image
    bodies = classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)

    # If no body is detected, return the original image
    if len(bodies) == 0:
        return gray

    # Draw a rectangle around the full body
    for (x, y, w, h) in bodies:
        cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Return the image with the full body rectangle drawn on it
    return gray



def save_detected_face(detected_face, landmarks, analyzed_dir, filename, symmetry_score, lip_fullness_score,
                       adiposity_score, eye_size_score, eye_distance, midline, fwhr_score, interocular_ratio,
                       nose_chin_lip_ratio, facial_thirds_ratio, show=True):
    """
    Saves the specified detected face in the specified analyzed directory with its corresponding beauty scores in the filename, as well as the midline of the face and landmarks, and displays it for 1 second.
    """
    # Format the beauty scores to two decimal places
    symmetry_score_str = "{:.2f}".format(symmetry_score)
    lip_fullness_score_str = "{:.2f}".format(lip_fullness_score)
    adiposity_score_str = "{:.2f}".format(adiposity_score)
    eye_size_score_str = "{:.2f}".format(eye_size_score)
    eye_distance_str = "{:.2f}".format(eye_distance)
    fwhr_score_str = "{:.2f}".format(fwhr_score)
    interocular_ratio_str = "{:.2f}".format(interocular_ratio)
    nose_chin_lip_ratio_str = "{:.2f}".format(nose_chin_lip_ratio)
    facial_thirds_ratio_str = "{:.2f}, {:.2f}, {:.2f}".format(*facial_thirds_ratio)

    # Build the new filename
    new_filename = f"{os.path.splitext(filename)[0]}_sym_{symmetry_score_str}_lip_{lip_fullness_score_str}_adi_{adiposity_score_str}_eye_{eye_size_score_str}_dist_{eye_distance_str}_mid_{midline[0]}-{midline[1]}-{midline[2]}-{midline[3]}_fwhr_{fwhr_score_str}_ior_{interocular_ratio_str}_ncl_{nose_chin_lip_ratio_str}_ftr_{facial_thirds_ratio_str}.jpeg"

    # Draw the midline and landmarks on the detected face
    for i in range(0, landmarks.num_parts):
        x = landmarks.part(i).x
        y = landmarks.part(i).y
        cv2.circle(detected_face, (x, y), 2, (255, 0, 0), -1)

    cv2.putText(detected_face, f"Symmetry: {symmetry_score_str}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(255, 255, 255), 1)
    cv2.putText(detected_face, f"Lip fullness: {lip_fullness_score_str}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(255, 255, 255), 1)
    cv2.putText(detected_face, f"Adiposity: {adiposity_score_str}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(255, 255, 255), 1)
    cv2.putText(detected_face, f"Eye size score: {eye_size_score_str}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(255, 255, 255), 1)
    cv2.putText(detected_face, f"Eye distance: {eye_distance_str}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(255, 255, 255), 1)
    cv2.putText(detected_face, f"FWHR score: {fwhr_score_str}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(255, 255, 255), 1)
    cv2.putText(detected_face, f"Interocular ratio: {interocular_ratio_str}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.9,(255, 255, 255), 1)
    cv2.putText(detected_face, f"Nose/Chin/Lip ratio: {nose_chin_lip_ratio_str}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX,0.9, (255, 255, 255), 1)
    cv2.putText(detected_face, f"Facial thirds ratio: {facial_thirds_ratio_str}", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 1)

    # Write the detected face to disk and display it for 1 second
    if not os.path.exists("scored_hon"):
        os.makedirs("scored_hon")

    # Write the detected face to disk and display it for 1 second
    try:
        save_path = os.path.join(path, new_filename)
        cv2.imwrite(save_path, detected_face)
        print(f"Detected face saved as {new_filename} in {save_path}.")

        if show:
            cv2.imshow('Detected Face', detected_face)
            cv2.waitKey(1000)  # delay for 1 second
            cv2.destroyAllWindows()
    except:
        print("Error saving detected face.")
        pass


from detectfun import *
import csv
from foldermg import organize_by_profile as og

# Define the directory paths
unclassified_dir = "/Users/peterpichler/Desktop/female"
scored_dir = "scored_hon"

# Create the face detector and landmark predictor
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Create the scored directory if it doesn't exist
if not os.path.exists(scored_dir):
    os.makedirs(scored_dir)

#write headers to csv before appending data

header_row = ['Image Filename', 'Profile Hash', 'Name', 'Picture Number', 'Attractiveness',
              'Symmetry Score', 'Adiposity Score', 'Midline', 'Lip Fullness Score',
              'Eye Distance', 'Eye Size Score', 'FWHR Score', 'Interocular Ratio',
              'Nose-Chin-Lip Ratio', 'Facial Thirds Ratio']

# write header row to CSV file
with open('scores_hon.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header_row)


# Iterate through the unclassified directory
for image_filename in os.listdir(unclassified_dir):

    try:

        # Ignore non-image files
        if not image_filename.endswith((".jpg", ".jpeg", ".png")):
            continue

        # open CSV file for appending

        #adapted to hot or not

        parts = image_filename.split('_')

        # extract the individual parts and assign them to variables


        profile_hash = parts[0]
        attractiveness = parts[1].rstrip(".jpg")
        name = "hon_candidate"
        picture_num = 1

        '''
        picture_num = int(parts[2])
        attractiveness = parts[3]
        attractiveness = os.path.splitext(attractiveness)[0]
        '''



        # Define the path to the image file
        img_path = os.path.join(unclassified_dir, image_filename)

        # Load the image
        img = cv2.imread(img_path)

        # Show the image
        #cv2.imshow("Original Image", img)

        # Detect faces in the image
        faces, gray = detect_faces(img, face_detector)

        # Skip the picture if no face is detected
        if len(faces) == 0:
            continue

        # Find the biggest detected face
        biggest_face = get_biggest_face(faces)

        landmarks = landmark_predictor(gray, biggest_face)

        # Compute the symmetry score of the detected face
        # Call the function to calculate beauty scores
        symmetry_score, lip_fullness_score, adiposity_score, eye_size_score, eye_distance, midline, fwhr_score, \
        interocular_ratio, nose_chin_lip_ratio, facial_thirds_ratio = calculate_scores(gray, biggest_face, landmarks)

        landmarks_array = np.empty([68, 2], dtype=int)
        for i in range(68):
            landmarks_array[i] = (landmarks.part(i).x, landmarks.part(i).y)

        # Define the path to the scored image file

        scored_path = os.path.join(scored_dir, image_filename)

        # Save the detected face with its corresponding beauty scores in the filename
        save_detected_face(detected_face=gray, landmarks=landmarks, analyzed_dir=scored_dir,
                           filename=image_filename,
                           symmetry_score=symmetry_score, adiposity_score=adiposity_score, midline=midline,
                           lip_fullness_score=lip_fullness_score, eye_distance=eye_distance,
                           eye_size_score=eye_size_score,
                           fwhr_score=fwhr_score, interocular_ratio=interocular_ratio,
                           nose_chin_lip_ratio=nose_chin_lip_ratio,
                           facial_thirds_ratio=facial_thirds_ratio, show=False)



        scores = [image_filename, profile_hash, name, picture_num, attractiveness, symmetry_score, adiposity_score, midline, lip_fullness_score,
                  eye_distance, eye_size_score, fwhr_score, interocular_ratio, nose_chin_lip_ratio,
                  facial_thirds_ratio]

        with open('scores_hon.csv', 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)

            # write scores to CSV file
            writer.writerow(scores)

    except Exception as e:
        print("Error occurred while processing file:", image_filename)
        print("Error message:", e)

import os
os.system("say 'all pictures scored'")


'''profile organizer has still to be written'''

# Replace 'filename.csv' with the actual file name and location
df = pd.read_csv('scores_hon.csv', sep= ",")

# Select columns with numeric data types and Profile Hash column
df = df.select_dtypes(include=[float, int]).assign(Profile_Hash=df['Profile Hash'])


# Convert the 'attractiveness' column from float to integer
df['Attractiveness'] = df['Attractiveness'].astype(int)

print(df.dtypes)


# Group the rows by Profile Hash and get the median for each column
#df_median = df.groupby('Profile_Hash').median()

# Calculate the correlation between each column and the Attractiveness column
corr = df.drop('Profile_Hash', axis=1).corrwith(df['Attractiveness'])

print(f"correlations\n{corr}")

# Group the data by attractiveness rating and calculate the median for each column
df = df.select_dtypes(include=[float, int])
df_grouped = df.groupby('Attractiveness').median()

# Visualize the median scores for each rating
plt.figure(figsize=(8,6))
#plt.plot(df_grouped.index, df_grouped.iloc[:,0], 'o-', label='Picture Number')
plt.plot(df_grouped.index, df_grouped.iloc[:,1], 'o-', label='Symmetry Score')
plt.plot(df_grouped.index, df_grouped.iloc[:,2], 'o-', label='Adiposity Score')
plt.plot(df_grouped.index, df_grouped.iloc[:,3], 'o-', label='Lip Fullness Score')
#plt.plot(df_grouped.index, df_grouped.iloc[:,4], 'o-', label='Eye Distance')
plt.plot(df_grouped.index, df_grouped.iloc[:,5], 'o-', label='Eye Size Score')
plt.plot(df_grouped.index, df_grouped.iloc[:,6], 'o-', label='FWHR Score')
#plt.plot(df_grouped.index, df_grouped.iloc[:,7], 'o-', label='Interocular Ratio')
plt.plot(df_grouped.index, df_grouped.iloc[:,8], 'o-', label='Nose-Chin-Lip Ratio')
plt.legend()
plt.xlabel('Attractiveness rating')
plt.ylabel('Median score')
plt.title('Median scores for each attractiveness rating')
plt.show()

for j in range(1, 9):

    df_filtered = df[df['Attractiveness'] == j]

    # Calculate the range of values for each column
    range_dict = {}
    for col in df_filtered.columns[:-1]:
        if col != 'Eye Distance' and col != 'Picture Number' and col !="Interocular Ratio" and col !="Attractiveness"\
                and col != "Profile Hash":
            col_min = df_filtered[col].min()
            col_max = df_filtered[col].max()
            range_dict[col] = (col_min, col_max)

    # Plot the range of values for each column
    colors = sns.color_palette('husl', n_colors=len(range_dict))
    plt.figure(figsize=(10, 8))
    for i, col in enumerate(range_dict):
        plt.hlines(y=i, xmin=range_dict[col][0], xmax=range_dict[col][1], color=colors[i], linewidth=7)
        plt.text(range_dict[col][1], i, f'{range_dict[col][1]:.2f}', fontsize=10, fontweight='bold')
    plt.yticks(range(len(range_dict)), range_dict.keys(), fontsize=10)
    plt.xlabel('Value range', fontsize=16)
    plt.title(
        f'Range of values for rating = {j} | {len(df_filtered)} ({len(df_filtered) / len(df) * 100:.2f}%)',
        fontsize=20, fontweight='bold')
    plt.grid(axis='x')
    plt.show()


X = df.drop(['Attractiveness'], axis=1)
y = df['Attractiveness']

# Replace infinite or very large values with NaN
X = X.replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.mean())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize a linear regression model and fit it to the training data
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model using the coefficient of determination (R-squared), mean squared error (MSE), and mean absolute error (MAE)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Print the performance metrics
print(f"R-squared score: {r2:.2f}")
print(f"Mean squared error: {mse:.2f}")
print(f"Mean absolute error: {mae:.2f}")




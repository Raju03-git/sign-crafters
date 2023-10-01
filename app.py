from flask import Flask, render_template, Response
import cv2
import time
import os
import requests

app = Flask(__name__)

# Create a folder to store frames if it doesn't exist
if not os.path.exists('Frames'):
    os.makedirs('Frames')

# Define the API endpoint and other parameters
endpoint = 'https://centralindia.api.cognitive.microsoft.com'
projectId = 'e178ed51-56f9-4882-b5ef-329114969d2d'
iterationId = 'Iteration4'
predictionKey = 'bfba0ebe4ca748119c16625f57f3453d'

# Define the base URL for the prediction API
api_url = f'{endpoint}/customvision/v3.0/prediction/{projectId}/classify/iterations/{iterationId}/image'

# Define headers
headers = {
    'Prediction-Key': predictionKey
}


# Define a global variable to store the latest frame
latest_frame = None

# Function to make the API call for prediction
def make_prediction(image_path):
    try:
        # Create a FormData-like object
        files = {'image': ('image.jpg', open(image_path, 'rb'), 'image/jpeg')}
        
        # Make the API call
        response = requests.post(api_url, headers=headers, files=files)

        if response.status_code == 200:
            predictions = response.json().get('predictions', [])
            if predictions:
                return predictions[0]['tagName']
            else:
                return "No prediction"
        else:
            print(f'API call failed with status code: {response.status_code}')
    except Exception as e:
        print(f'API call failed: {e}')
    return "Error"

def video_stream():
    global latest_frame  # Access the global variable
    # Open the camera (0 represents the default camera, change if necessary)
    cap = cv2.VideoCapture(0)

    # Initialize a timer
    last_frame_time = time.time()
    prediction_result = "No prediction yet"
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Calculate the time since the last frame
        current_time = time.time()
        elapsed_time = current_time - last_frame_time
        
        # If 5 seconds have passed, save the frame and update the timer
        if elapsed_time >= 5:
            frame_filename = os.path.join('Frames', f'frame_{current_time}.jpg')
            cv2.imwrite(frame_filename, frame)
            
             # Update the global variable with the latest frame
            latest_frame = frame_filename

            # Make the API call for prediction
            prediction_result = make_prediction(frame_filename)
            print('Prediction Result:', prediction_result)
            
            last_frame_time = current_time

        # Convert the frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
                   b'Prediction: ' + prediction_result.encode() + b'\r\n')

@app.route('/')
def index():
    # Pass an initial prediction result (e.g., "No prediction yet") to the template
    initial_prediction = "No prediction yet"
    return render_template('index.html', prediction_result=initial_prediction)

@app.route('/video_feed')
def video_feed():
    return Response(video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    global latest_frame  # Access the global variable
    if latest_frame:
        prediction_result = make_prediction(latest_frame)
        return prediction_result
    else:
        return "No prediction yet"

if __name__ == '__main__':
    app.run(debug=True)
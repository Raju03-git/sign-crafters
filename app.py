from flask import Flask, render_template, Response
import cv2
import time
import os
import requests

app = Flask(__name__)

if not os.path.exists('Frames'):
    os.makedirs('Frames')

endpoint = 'https://signlanguageassistant-prediction.cognitiveservices.azure.com'
projectId = '3fa73b26-9b74-4fdc-a786-4d6570286e5b'
iterationId = 'Iteration2'
predictionKey = 'a25c47fce3f74281a659819b495c00dc'

api_url = f'{endpoint}/customvision/v3.0/prediction/{projectId}/classify/iterations/{iterationId}/image'
headers = {
    'Prediction-Key': predictionKey
}

latest_frame = None
processing_enabled = False
def make_prediction(image_path):
    try:
        files = {'image': ('image.jpg', open(image_path, 'rb'), 'image/jpeg')}
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
    global latest_frame
    cap = cv2.VideoCapture(0)
    last_frame_time = time.time()
    prediction_result = ""
    while True:
        success, frame = cap.read()
        if not success:
            break

        current_time = time.time()
        elapsed_time = current_time - last_frame_time

        if elapsed_time >= 5 and processing_enabled:
            frame_filename = os.path.join('Frames', f'frame_{current_time}.jpg')
            cv2.imwrite(frame_filename, frame)

            latest_frame = frame_filename

            prediction_result = make_prediction(frame_filename)
            print('Prediction Result:', prediction_result)

            last_frame_time = current_time

        ret, buffer = cv2.imencode('.jpg', frame)
        if ret:
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
                   b'Prediction: ' + prediction_result.encode() + b'\r\n')

@app.route('/')
def index():
    initial_prediction = ""
    return render_template('index.html', prediction_result=initial_prediction)

@app.route('/video_feed')
def video_feed():
    return Response(video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    global latest_frame
    if latest_frame:
        prediction_result = make_prediction(latest_frame)
        return prediction_result
    else:
        return "No prediction yet"

@app.route('/start_processing')
def start_video_processing():
    global processing_enabled
    processing_enabled = True
    return "Video processing started"

@app.route('/stop_processing')
def stop_video_processing():
    global processing_enabled
    processing_enabled = False
    return "Video processing stopped"

if __name__ == '__main__':
    app.run(debug=True)

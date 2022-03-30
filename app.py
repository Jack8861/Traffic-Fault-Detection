from flask import Flask, render_template, Response, request
import time
import cv2 as cv
# from rules import TrafficRules


app = Flask(__name__)

STATIC = r"/home/rahim/Desktop/traffic_fault/static"
# rule = TrafficRules()


@app.route('/')
def home1():
    return render_template('index.html')


def run(path):
    cap = cv.VideoCapture(path)
    while (cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            frame = cv.resize(frame, (0, 0), fx=0.5, fy=0.5)
            img = cv.imencode('.jpg', frame)[1].tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')
            # rule.check_rules(frame=frame)
            time.sleep(0.05)
        else:
            break
    img = cv.imread(
        r"/home/rahim/Desktop/traffic_fault/static/images/image.jpg")
    img = cv.imencode('.jpg', img)[1].tobytes()
    yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')


@app.route('/showcase/<string:vid>', methods=['GET', 'POST'])
def showcase(vid):
    print(vid)
    path = r"/home/rahim/Desktop/traffic_fault/static/videos/{}.mp4".format(
        vid)
    return Response(run(path),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)

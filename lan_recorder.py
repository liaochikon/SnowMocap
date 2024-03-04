import json
import pickle
import socket
import time
import cv2

name = "katana2"

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out0 = cv2.VideoWriter("video/" + name + str(0) + '.avi', fourcc, 30.0, (1280, 720))
out1 = cv2.VideoWriter("video/" + name + str(1) + '.avi', fourcc, 30.0, (1280, 720))
out2 = cv2.VideoWriter("video/" + name + str(2) + '.avi', fourcc, 30.0, (1280, 720))
out3 = cv2.VideoWriter("video/" + name + str(3) + '.avi', fourcc, 30.0, (1280, 720))

ip="192.168.50.237"
Ports = [6666 , 6667, 6668, 6669]
Socs = []
for port in Ports:
    soc=socket.socket(socket.AF_INET , socket.SOCK_DGRAM)
    soc.bind((ip, port))
    Socs.append(soc)

img_shape = (720, 1280)

frame_deltatime = []

while True:
    a = time.time()
    img0 = cv2.imdecode(pickle.loads(Socs[0].recvfrom(1000000)[0]), cv2.IMREAD_COLOR)
    img1 = cv2.imdecode(pickle.loads(Socs[1].recvfrom(1000000)[0]), cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(pickle.loads(Socs[2].recvfrom(1000000)[0]), cv2.IMREAD_COLOR)
    img3 = cv2.imdecode(pickle.loads(Socs[3].recvfrom(1000000)[0]), cv2.IMREAD_COLOR)
    cv2.imshow('frame', cv2.resize(img0, (800, 450)))
    if cv2.waitKey(1) == ord('q'):
        break
    out0.write(img0)
    out1.write(img1)
    out2.write(img2)
    out3.write(img3)

    deltatime = time.time() - a
    frame_deltatime.append(deltatime)
    print("FPS ", 1 / (deltatime))

with open("video/" + name + ".json" , "w") as outfile:
    jsonStr = json.dumps(frame_deltatime)
    outfile.write(jsonStr)

out0.release()
out1.release()
out2.release()
out3.release()
cv2.destroyAllWindows()
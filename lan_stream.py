from socket import socket
from multiprocessing import Process
import cv2, socket, pickle, os
import numpy as np

s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1000000)
ismain=True
server_ip = "192.168.50.145"
server_port = 6666
scale = 0.4

cap = [cv2.VideoCapture(0), cv2.VideoCapture(3), cv2.VideoCapture(1), cv2.VideoCapture(2)]
for c in cap:
    c.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

def udp(frame, num):
    ret, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 30])
    x_as_byte = pickle.dumps(buffer)
    s.sendto((x_as_byte), (server_ip, server_port + num))

def send_frame(num):
    while True:
        print(num)
        _, frame = cap[num].read()
        udp(frame, num)
        cv2.imshow(str(num), frame)
        if cv2.waitKey(1) == ord('q'):
            print("stop")
            break
 
if __name__ == '__main__':
    frame0 = Process(target=send_frame, args=(0,))
    frame1 = Process(target=send_frame, args=(1,))
    frame2 = Process(target=send_frame, args=(2,))
    frame3 = Process(target=send_frame, args=(3,))

    frame0.start()
    frame1.start()
    frame2.start()
    frame3.start()

    frame0.join()
    frame1.join()
    frame2.join()
    frame3.join()

    for c in cap:
        c.release()
    cv2.destroyAllWindows()

from pyfirmata2 import Arduino, SERVO
from time import sleep

board = Arduino(Arduino.AUTODETECT)

indexFlex = 2
indexExtend = 3
middleFlex = 4
middleExtend = 5
ringFlex = 6
ringExtend = 7
pinkyFlex = 8
pinkyExtend = 9
thumbFlex = 10
thumbRotate = 11

for i in range(2, 12):
    board.digital[i].mode = SERVO

def rotateServo(servo, angle):
    board.digital[servo].write(angle)

for i in range(10):
    rotateServo(i+2,5)

sleep(5)
for i in range(10):
    print(i+2)
    rotateServo(i+2,100)
    sleep(1)
sleep(2)

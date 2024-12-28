
import RPi.GPIO as GPIO
from time import sleep

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(3, GPIO.OUT)
pwm=GPIO.PWM(3, 50)
pwm.start(0)

def SetAngle(angle):
	""" Turns the servo by a certain angle

	Args:
		angle (int): desired angle for the servo to turn
	"""
	duty = angle / 18 + 2
	GPIO.output(3, True)
	pwm.ChangeDutyCycle(duty)
	sleep(1)
	GPIO.output(3, False)
	pwm.ChangeDutyCycle(0)

def servo1():
	""" Calls SetAngle, moving the servo motor by 90 degrees, then back 90 degrees
	"""
    SetAngle(90)
    SetAngle(180)
    SetAngle(90)
    pwm.stop()
    GPIO.cleanup()

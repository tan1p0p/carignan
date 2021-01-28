import multiprocessing
from multiprocessing import Value
import time

import cv2
import numpy as np
import serial
from serial.tools import list_ports


class LaserHandler():
    def __init__(self):
        self.serial_power = 60000
        self.serial_seconds = 0.5
        self.is_shooting = Value('b', False)
        ser = self.__select_port()
        if ser == 'dummy':
            self.ser = None
        else:
            self.ser = ser
            self.__show_connection(self.ser)

    def __select_port(self):
        ser = serial.Serial()
        ser.baudrate = 9600
        ser.timeout = 0.1
        ports = list_ports.comports()
        devices = [info.device for info in ports]

        if len(devices) == 0:
            print('error: device not found')
            return None
        elif len(devices) == 1:
            print('only found %s' % devices[0])
            ser.port = devices[0]
        else:
            for i in range(len(devices)):
                print(f'{i:3d}: open {devices[i]}')
            print(f'{i+1:3d}: use dummy device for developing')
            print('input number of target port >> ', end='')
            num = int(input())
            if num == i+1:
                return 'dummy'
            ser.port = devices[num]

        try:
            ser.open()
            return ser
        except:
            print('error when opening serial')
            return None

    def __show_connection(self):
        for _ in range(3):
            self.ser.write(str(60000).encode('utf-8'))
            self.ser.write(b'\n')
            self.ser.reset_output_buffer()
            time.sleep(0.5)
            
            self.ser.write(str(0).encode('utf-8'))
            self.ser.write(b'\n')
            self.ser.reset_output_buffer()
            time.sleep(0.5)

    def shoot_laser(self, ser, power, seconds):
        def shoot(is_shooting):
            if is_shooting.value == 0:
                is_shooting.value = 1
                self.ser.write(str(power).encode('utf-8'))
                self.ser.write(b'\n')
                self.ser.reset_output_buffer()
                time.sleep(seconds)

                self.ser.write(str(0).encode('utf-8'))
                self.ser.write(b'\n')
                self.ser.reset_output_buffer()
                is_shooting.value = 0

        def dummy(is_shooting):
            if is_shooting.value == 0:
                is_shooting.value = 1
                print('laser on')
                time.sleep(seconds)
                print('laser off')
                is_shooting.value = 0

        if not ser is None:
            p = multiprocessing.Process(target=shoot, args=(self.is_shooting,))
        else:
            p = multiprocessing.Process(target=dummy, args=(self.is_shooting,))
        p.start()


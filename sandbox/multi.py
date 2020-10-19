from multiprocessing import Array, Process
import socket
import time

def caiman(array):
    i = 0
    while True:
        print('online CNMF-E is running...')
        array.append(i)
        time.sleep(1)
        i += 1

def server(ip, port, array):
    print('now listening...')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((ip, port))
        s.listen(1)
        while True:
            conn, addr = s.accept()
            with conn:
                while True:
                    data = conn.recv(1024) # データ受け取り
                    if not data:
                        break
                    print(f'data : {data}, addr: {addr}, array: {array}')
                    conn.sendall(b'Received: ' + data) # クライアントにデータを返す(b -> byte でないといけない)

if __name__ == '__main__':
    array = Array('i', 15)
    server = Process(target=server, args=('127.0.0.1', 50007, array))
    caiman = Process(target=caiman, args=(array))
    server.start()
    caiman.start()
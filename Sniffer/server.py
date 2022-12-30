#!/usr/bin/env python
# -*-coding:utf-8 -*-
import socket
import struct
import json
import os

def main():
    download_directory = os.getcwd()+'/traffic_collect'
    HOST = '192.168.56.13'
    PORT = 9091
    unlock = '1'
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(5)

    while True:
        print("Waiting for connection......")
        connection, address = server.accept()
        print("Connection success......")
        while True:
            try:
                print("Wait for receiving traffic data")
                # Receive file numbers
                receive = connection.recv(8096)
                if not receive: break

                # Parse the command
                command = receive.decode('utf-8').split()
                file_number = command[0]
                    
                for i in range(int(file_number)):
                    # Open a new file as writing, then write the new file from the server host
                    # First step: receive the header size
                    object = connection.recv(4)
                    header_size = struct.unpack('i', object)[0]

                    # Second step: receive the header
                    header_bytes = connection.recv(header_size)

                    # Third step: extract the detail info. of the real data from header
                    header_json = header_bytes.decode('utf-8')
                    header_dic = json.loads(header_json)
                    '''
                    header_dic = {
                        'filename': filename,
                        'file_size': os.path.getsize('%s/%s' % (shared_directory, filename))
                    }
                    '''
                    total_size = header_dic['file_size']
                    file_name = header_dic['filename']
                    print(total_size)
                    print(download_directory)
                    print(file_name)

                    # Fourth step: receive the real data
                    connection.send(unlock.encode('utf-8'))
                    try:
                        f = open('%s/%s' % (download_directory, file_name), 'wb')
                        recv_size = 0
                        while recv_size < total_size:
                            line = connection.recv(1024)
                            f.write(line)
                            recv_size += len(line)
                            print('Total size: %s, Already downloads: %s' % (total_size, recv_size))
                        print('Finish download file')
                        f.close()
                        connection.send(unlock.encode('utf-8'))

                    except:
                        print('Fail to open the file')

            except Exception:
                print(Exception)
                connection.close()
                break
    server.close()

if __name__ == "__main__":
    main()
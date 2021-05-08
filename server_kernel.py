
from ipykernel.kernelbase import Kernel
import numpy as np
import os
import matplotlib.pyplot as plt
from io import BytesIO
import urllib, base64
import socket

class server_kernel(Kernel):
    implementation = 'server'
    implementation_version = '1.0'
    language = 'python'  # will be used for
                         # syntax highlighting
    language_version = '3.6'
    language_info = {'name': 'server_kernel',
                     'mimetype': 'text/plain',
                     'extension': '.py'}
    banner = "Simple client"

    def do_execute(self, code, silent,
                   store_history=True,
                   user_expressions=None,
                   allow_stdin=False):

        port= code.split('\n')[0]
        file_name = code.split('\n')[1]
        
        
        s = socket.socket()         # Create a socket object
        host = socket.gethostname() # Get local machine name
        port = int(port)           # Reserve a port for your service.
        s.bind((host, port))        # Bind to the port
        
        
        s.listen(5)  
        result=""
        # Now wait for client connection.
        while True:
            c, addr = s.accept()     # Establish connection with client.
            result += str('Got connection from')+str(addr)
            
            file1 = open(file_name,"r+") 
            data=file1.read()
            c.send(data.encode())
            
            c.close()   
        
        if not silent:
            # We send the standard output to the
            # client.
            self.send_response(
                self.iopub_socket,
                'stream', {
                    'name': 'stdout',
                    'data': ('Plotting {n} ')})

            # We prepare the response with our rich
            # data (the plot).
            content = {
                'source': 'kernel',

                # This dictionary may contain
                # different MIME representations of
                # the output.
                'data': {
                        'text/plain' : result
                },

                
            }

            # We send the display_data message with
            # the contents.
            self.send_response(self.iopub_socket,
                'display_data', content)

        # We return the exection results.
        return {'status': 'ok',
                'execution_count':
                    self.execution_count,
                'payload': [],
                'user_expressions': {},
               }

if __name__ == '__main__':
    from ipykernel.kernelapp import IPKernelApp
    IPKernelApp.launch_instance(
        kernel_class=server_kernel)


from ipykernel.kernelbase import Kernel
from gingerit.gingerit import GingerIt
from io import BytesIO
import urllib, base64
import string
import sys

# reading the text file
# This functio will return a
# list of the lines of text
# in the file.
def read_file(filename):

    try:
        with open(filename, 'r') as f:
            data = f.read()
        return data

    except IOError:
        print("Error opening or reading input file: ", filename)
        sys.exit()

class grammarKernel(Kernel):
    implementation = 'Grammar'
    implementation_version = '1.0'
    language = 'python'  # will be used for
                         # syntax highlighting
    language_version = '3.6'
    language_info = {'name': 'grammar',
                     'mimetype': 'text/plain',
                     'extension': '.py'}
    banner = "Simple Grammar Check"

    def do_execute(self, code, silent,
                   store_history=True,
                   user_expressions=None,
                   allow_stdin=False):
        file_path= code.split("\n")[0];
        file_Data=read_file(file_path);

        parser = GingerIt()
        fileData='The output file has been created with the name correctGrammar.txt.';
        with open('correctGrammar.txt', 'w') as f:
            f.write(parser.parse(file_Data)['result'])
        
        
        
    
        if not silent:
            # We send the standard output to the
            # client.
            self.send_response(
                self.iopub_socket,
                'stream', {
                    'name': 'stdout',
                    'data': ('Correcting grammmar in file')})

            # We prepare the response with our rich
            # data (the plot).
            content = {
                'source': 'kernel',

                # This dictionary may contain
                # different MIME representations of
                # the output.
                'data': {
                    'text/plain': fileData
                },

                # We can specify the image size
                # in the metadata field.
               
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
        kernel_class=grammarKernel)

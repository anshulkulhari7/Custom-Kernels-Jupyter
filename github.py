
from ipykernel.kernelbase import Kernel
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PyGithub.github.MainClass import Github
import urllib, base64
 
def _to_png(fig):
    """Return a base64-encoded PNG from a
    matplotlib figure."""
    imgdata = BytesIO()
    fig.savefig(imgdata, format='png')
    imgdata.seek(0)
    return urllib.parse.quote(
        base64.b64encode(imgdata.getvalue()))

_numpy_namespace = {n: getattr(np, n)
                    for n in dir(np)}

def _parse_function(code):
    """Return a NumPy function from a
    string 'y=f(x)'."""
    return lambda x: eval(code.split('=')[1].strip(),
                          _numpy_namespace, {'x': x})


def get_repos():
    # Then play with your Github objects:
    repo_list = ""
    for repo in g.get_user().get_repos():
        repo_list +=repo.name +"\n"
    return repo_list


class github(Kernel):
    implementation = 'Github'
    implementation_version = '1.0'
    language = 'python'  # will be used for
                         # syntax highlighting
    language_version = '3.6'
    language_info = {'name': 'github',
                     'mimetype': 'text/plain',
                     'extension': '.py'}
    banner = "Simple github"

    def do_execute(self, code, silent,
                   store_history=True,
                   user_expressions=None,
                   allow_stdin=False):

        
        lines= code.split('\n')
        token = lines[0]
        g=Github(token)
        
        user = g.get_user()
        
        function = line[1].split(':')
        function_name = function[0]
        result=""
        if function_name=="get_user_repos":
            result=get_repos()
        
        if not silent:
            # We send the standard output to the
            # client.
            self.send_response(
                self.iopub_socket,
                'stream', {
                    'name': 'stdout',
                    'data': ('Plotting {n} '
                             'function(s)'). \
                            format(n=len(functions))})

            # We prepare the response with our rich
            # data (the plot).
            content = {
                'source': 'kernel',

                # This dictionary may contain
                # different MIME representations of
                # the output.
                'data': {
                    'text/plain': result
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
        kernel_class=github)

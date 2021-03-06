import os
from distutils.command.build_ext import build_ext
from distutils.core import setup, Extension
from zipfile import ZipFile

import numpy as np
from Cython.Build import cythonize


class QPBOInstall(build_ext):
    def run(self):
        # locate urlretrieve: with Python3, this has been moved to urllib.request
        import urllib
        if hasattr(urllib, "urlretrieve"):
            urlretrieve = urllib.urlretrieve
        else:
            import urllib.request
            urlretrieve = urllib.request.urlretrieve
            # fetch and unpack the archive. Not the nicest way...
        urlretrieve("http://pub.ist.ac.at/~vnk/software/QPBO-v1.4.src.zip",
                    "QPBO-v1.4.src.zip")
        zfile = ZipFile("QPBO-v1.4.src.zip")
        zfile.extractall('.')
        build_ext.run(self)


qpbo_directory = "QPBO-v1.4.src"

files = ["QPBO.cpp", "QPBO_extra.cpp", "QPBO_maxflow.cpp",
         "QPBO_postprocessing.cpp"]

files = [os.path.join(qpbo_directory, f) for f in files]
files.insert(0, "src/pyqpbo.pyx")

setup(name='pyqpbo',
      packages=['pyqpbo'],
      version="0.1.2",
      author="Andreas Mueller",
      author_email="t3kcit@gmail.com",
      description='QPBO interface and alpha expansion for Python',
      url="http://pystruct.github.io",
      cmdclass={"build_ext": QPBOInstall},
      ext_modules=cythonize(
          [Extension('pyqpbo.pyqpbo', sources=files, language='c++',
                     include_dirs=[qpbo_directory, np.get_include()],
                     library_dirs=[qpbo_directory],
                     extra_compile_args=["-fpermissive", "-O3"])
           ])
      )

import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
setuptools.setup(
     name='QGOpt',  
     version='0.11',
     author="I. Luchnikov, M. Krechetov",
     author_email="luchnikovilya@gmail.com",
     description="Riemannian optimization for quantum mechanics",
     url="https://github.com/LuchnikovI/QGOpt",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
 )
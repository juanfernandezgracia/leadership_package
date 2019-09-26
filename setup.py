from distutils.core import setup

setup(
     name='leadership_KS',
     version='0.0.1',
     author='Juan Fernandez Gracia',
     author_email='juanfernandez1984@gmail.com',
     packages=['leadership_KS'],
     scripts=[],
     url='http://github.com/IFISC/leadership_KS',
     license='LICENSE.txt',
     description='A package for detecting follower-followee relations in point (acoustic) data.',
     # long_description=open('README.md').read(),
     # long_description_content_type="text/markdown",
     install_requires=[
         # "python >= 3.7.0",
         "numpy >= 1.16.0",
         'datetime',
         'scipy >= 1.2.0',
         # 'random'
     ],
     classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

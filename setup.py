from __future__ import absolute_import
from setuptools import setup
from setuptools import find_packages

setup(name='elephas',
      version='0.4.3',
      description='Deep learning on Spark with Keras',
      url='http://github.com/maxpumperla/elephas',
      download_url='https://github.com/maxpumperla/elephas/tarball/0.4.3',
      author='Max Pumperla',
      author_email='max.pumperla@googlemail.com',
      install_requires=['cython', 'tensorflow==1.15.4', 'keras==2.2.4', 'hyperas', 'flask', 'six', 'pyspark==2.4.5'],
      extras_require={
        'java': ['pydl4j>=0.1.3'],
        'tests': ['pytest', 'pytest-pep8', 'pytest-cov', 'mock']
    },
      packages=find_packages(),
      license='MIT',
      zip_safe=False,
      classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Environment :: Console',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3'
    ])

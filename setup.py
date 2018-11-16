from __future__ import absolute_import
from setuptools import setup
from setuptools import find_packages

setup(name='elephas',
      version='0.4',
      description='Deep learning on Spark with Keras',
      url='http://github.com/maxpumperla/elephas',
      download_url='https://github.com/maxpumperla/elephas/tarball/0.4',
      author='Max Pumperla',
      author_email='max.pumperla@googlemail.com',
      install_requires=['cython', 'keras', 'tensorflow', 'hyperas', 'flask', 'six', 'pyspark', 'pydl4j>=0.1.3'],
      packages=find_packages(),
      license='MIT',
      zip_safe=False)

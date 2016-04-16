from setuptools import setup
from setuptools import find_packages

setup(name='elephas',
      version='0.3',
      description='Deep learning on Spark with Keras',
      url='http://github.com/maxpumperla/elephas',
      download_url='https://github.com/maxpumperla/elephas/tarball/0.3',
      author='Max Pumperla',
      author_email='max.pumperla@googlemail.com',
      install_requires=['keras', 'hyperas', 'flask'],
      license='MIT',
      packages=find_packages(),
      zip_safe=False)

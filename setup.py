from setuptools import setup

setup(name='elephas',
      version='0.2',
      description='Deep learning on Spark with Keras',
      url='http://github.com/maxpumperla/elephas',
      download_url = 'https://github.com/maxpumperla/elephas/tarball/0.1',
      author='Max Pumperla',
      author_email='max.pumperla@googlemail.com',
      install_requires=['keras'],
      license='MIT',
      packages=['elephas'],
      zip_safe=False)

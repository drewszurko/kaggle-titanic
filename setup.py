from setuptools import setup, find_packages

REQUIRED_PACKAGES = ['keras==2.1.3', 'h5py==2.7.1', 'xgboost']

# Setup parameters for GC ML Engine.
setup(name='trainer',
      version='0.1',
      install_requires=REQUIRED_PACKAGES,
      packages=find_packages(),
      include_package_data=True,
      description='Train Keras model on Google Cloud ML Engine.',
      author='Drew Szurko',
      license='MIT',
      zip_safe=False
      )

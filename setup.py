import os
from setuptools import setup

name = 'uit_scripts'

with open('README.md') as f:
    long_description = f.read()

here = os.path.abspath(os.path.dirname(__file__))

setup(name=name,
      description='Python scripts used by the fusion energy group at UiT The Arctic University of Norway.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      url='https://github.com/audunth/uit_scripts/tree/master/uit_scripts',
      author='Audun Theodorsen',
      author_email='audun.theodorsen@uit.no',
      license='MiT',
      version='1.0',
      packages=['uit_scripts'],
      python_requires='>=3.0',
      install_requires=['numpy>=1.15.0',
      			'scipy>=1.4.0',
			'mpmath>=1.0.0',
            'tqdm>=4.50.2'],
      classifiers=[
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Visualization',
      ],
      zip_safe=False)

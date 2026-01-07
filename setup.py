from setuptools import find_packages, setup

setup(
    name='student_score_prediction',
    version='0.1.0',
    author='Ilya Sevastyanov',
    author_email='sevastyanovilya@gmail.com',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'seaborn',
        'matplotlib',
        'scikit-learn',
        'catboost',
        'Flask',
        'dill',
    ]
)

from setuptools import find_packages, setup

setup(
    name='automl',
    packages=find_packages(include=['automl', 'automl.wrappers']),
    install_requires=[
        'numpy',
        'pandas',
        'statsmodels',
        'scikit-learn>=0.23',
        'lightgbm',
        'torch',
        'pytorch-forecasting',
        'pytorch_lightning',
        'tqdm'
    ],
    version='0.1.0',
    description='Auto machine learning project with focus on predict time series.',
    author='netlab',
    license='MIT',
)

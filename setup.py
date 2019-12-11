from setuptools import setup

setup(
    name='mipego4ml',
    version='0.1',
    packages=['mipego4ml', 'mipego4ml.mipego', 'mipego4ml.mipego.optimizer'],
    url='hyperparameter.ml',
    license='',
    author='Duc Anh Nguyen, Hao Wang, Thomas Back',
    author_email='d.a.nguyen@liacs.leidenuniv.nl',
    description='This project is a extension of MIP-EGO project',
    install_requires=['pandas', 'numpy', 'scipy', 'scikit-learn', 'joblib', 'dill']

)

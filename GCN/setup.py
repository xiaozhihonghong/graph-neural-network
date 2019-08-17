from setuptools import setup
from setuptools import find_packages

setup(name='pygcn',  # 生成的包的名字
      version='0.1',  # 版本号
      description='Graph Convolutional Networks in PyTorch',  # 包的简要描述
      author='Thomas Kipf',  # 包的作者
      author_email='thomas.kipf@gmail.com',   # 作者的邮箱地址
      url='https://tkipf.github.io',  # 程序的官网地址
      download_url='https://github.com/tkipf/pygcn',   # 程序的下载地址
      license='MIT',  # 程序的授权信息
      install_requires=['numpy',   # 需要安装的依赖包
                        'torch',
                        'scipy'
                        ],
      package_data={'pygcn': ['README.md']},
      packages=find_packages())
      # fine_packages()函数默认在和setup.py同一目录下搜索各个含有__init__.py的包
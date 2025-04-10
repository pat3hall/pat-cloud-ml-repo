from setuptools import setup, find_packages

# load contents of 'READM.rst' file to "readme" string
#  note: readlines() creates a list of the lines while "readme()" creates 1 string
with open('README.rst', encoding='UTF-8') as f:
    readme = f.read()


# call setup() method , Notes:
#   - install_requires[<pkg list>] currently boto3
#   - find_packages('scr') will recursively find '__init__.py' under 'src' directory to find packages
#   - package_dir{'': 'src'} a dict that points from and empty string to 'src' and is a
#     legacy thing disutils. This is needed because of 'find_package() is explicity searching 'src'
# add entry_point to define console scripts with '<scriptName> = <packageName>.<moduleName>:<functionName>'
setup(
        name='pgbackup',
        version='0.1.0',
        description='Database backups locally or to AWS S3.',
        long_description=readme,
        author='Pat',
        author_email='phreg20@gmail.com',
        install_requires=['boto3'],
        packages=find_packages('src'),
        package_dir={'': 'src'},
        entry_points={
            'console_scripts': [
                'pgbackup=pgbackup.cli:main',
            ],
        }


)


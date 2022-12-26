import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='tsaoptimized',
    version='0.1.0',
    author='Mike Huls',
    author_email='mike_huls@hotmail.com',
    description='Testing installation of Package',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/mike-huls/toolbox',
    # project_urls = {
    #     "name": "https://github.com/user/issues"
    # },
    license='MIT',
    packages=['tsaoptimized'],
    install_requires=['numpy','pandas','scikit-learn','tensorflow'],
)
Contributing

Contributions are always welcome, no matter how large or small! Before contributing, please read the code of conduct.

Not sure where to start?
When you feel ready to jump into the source code, a good place to start is to look for issues tagged with help wanted and/or good first issue.

Setup
install python v3
install pip
intall tensorflow v2

setup pytest as test runner in your IDE
setup crlf in git config


How to run tests
execute from command line command pytest

Pull Request Checklist

Fork the repository to your GitHub Account.
Clone forked repo to your local machine
Create your feature branch
Make code/documantation changes
Create pull request

Before sending your pull requests, make sure you followed this list.

Read contributing guidelines.
Read Code of Conduct.
Ensure you have signed the Contributor License Agreement (CLA).
Check if my changes are consistent with the guidelines.
Changes are consistent with the Coding Style.
Run Unit Tests.

Python coding style
Changes to TensorFlow Python code should conform to Google Python Style Guide

Use pylint to check your Python changes. To install pylint and check a file with pylint against TensorFlow's custom style definition:

pip install pylint
pylint --rcfile=tensorflow/tools/ci_build/pylintrc myfile.py
Note pylint --rcfile=tensorflow/tools/ci_build/pylintrc should run from the top level tensorflow directory.



<h1 align="center">AI Data Matching</h1>

<!-- GETTING STARTED -->
## Getting Started

1. Clone the official repo [neural_networks](https://github.com/educep/neural_networks)

   1. Create, activate your environment and, install the libraries.\
      In Windows run:
       ```sh
      .\ws_install_venv.bat
       ```
      In Linux run:
       ```sh
      .\lx_install_venv.sh
       ```

3. Create .env file in the root level
    ```sh
    LOCAL_PATH="path/to/your/file.xlsx
    ```

Remark that to run this project locally, in Windows you can use:

```sh
run_st_app.bat
```

5. Install precommit and test it in commit
  ```sh
  pre-commit install
  ```

9. Associate and push with the private github repo
```sh
git init
git commit -m "first commit"
git remote rm origin
git remote add origin https://github.com/git_user/my_repo.git
git push -u origin master
```


<!-- CONTRIBUTING -->
## Contributing

Any contributions you make are **greatly appreciated**, but to facilitate the revision, they must respect the current versioning policy described here below:
### Set-up
1. Pull last changes from master (`git pull origin master`)
2. Create a Feature branch (`git checkout -b feature/AmazingFeature`) or a bug fix branch (`git checkout -b bugfix/AmazingBugFix`)
3. Be sure to use linters as black and install pre-commit & black if needed (cf. How to section - precommit using at least the .yaml described)

### Developpment
1. Code your feature of bugfix
2. Create or update always your .gitignore/requirements.txt files
3. Adapt or create tests files related yo your development
4. ~~Test there is no code regression~~ (not implemented yet)
6. Git add your changes (`git add . `)
7. Commit your Changes (`git commit -m 'Add some AmazingFeature'`) verifying it uses pre-commit
8. Verify that you have last changes from master in your current branch (`git pull origin master`) to avoid conflicts with latest developments
10. Push to the Branch (`git push origin feature/AmazingFeature`)
11. Open a Pull Request when everything is ended
<br><br>

## Code structure
Current version at:


# How to

In the following section, there is small documentation to go faster in the deployment and good coding practices.
<br><br>

## Virtual environments

Please always use environments and adapt to be inline with requirements.txt

1. Create a new environment & activate
   ```sh
     # Windows
   > python -m venv .venv
   > Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   > .venv\Scripts\Activate.ps1
   (.venv) >

   # macOS
   % python3 -m venv .venv
   % source .venv/bin/activate
   (.venv) %
   ```

2. Deactivate current environment
     ```sh
        # Windows
      (.venv) > deactivate
      >
      # macOS
      (.venv) % deactivate
      %
     ```
<br/><br/>


## .gitignore

Adding a .gitignore file is a must to modern development and to be able to contribute between larger teams. There are a lot of templates and it should be updated if needed, but a standard we'll use for this project is the following:

```sh
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# Django stuff:
*.log
# local_settings.py
db.sqlite3
db.sqlite3-journal

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Media files
media/
# Elastic Beanstalk Files
.elasticbeanstalk/*
!.elasticbeanstalk/*.cfg.yml
!.elasticbeanstalk/*.global.yml

node_modules
*.json
*.config.js

#IDE
.idea/
.DS_Store
```

### To remove the directory from git repo

```sh
git rm -r one-of-the-directories
```
## Linters

In order to meet the best code-quality standards, you should use linters every time before you commit to ensure good quality standards.

A good article describing the benefits of linters is the following.\
[Link to article](https://sourcelevel.io/blog/what-is-a-linter-and-why-your-team-should-use-it)

In our project, we will mainly use Black but others linters could be added. Linters and using pre-commit is strongly recommended to automate the checks.
A good extensive article describing interactions is the following:\
[Link to Document](https://codeburst.io/tool-your-django-project-pre-commit-hooks-e1799d84551f
)
<br><br>

## Black

Black is a Python code formatter. It will reformat your entire file in place

according to the Black code style, which is pretty close to PEP8.

To quote the project README:

> Black is the uncompromising Python code formatter. By using it, you agree to
> cede control over minutiae of hand-formatting. In return, Black gives you speed,
> determinism, and freedom from `pycodestyle` nagging about formatting. You will
> save time and mental energy for more important matters.

- Configuration file: `pyproject.toml`
- Editor/IDE configuration: https://black.readthedocs.io/en/stable/editors.html
<br><br>

<!-- test -->
## Pre-commit

[pre-commit](https://github.com/pre-commit/pre-commit) is a Python framework for git hook management — we’ll use it to run Black against every commit you make to your project.

To configurate easily:
https://medium.com/gousto-engineering-techbrunch/automate-python-code-formatting-with-black-and-pre-commit-ebc69dcc5e03

Then, adding to run automatically Django tests, our **.pre-commit-config.yaml** file should be the following
```sh
repos:
    - repo: https://github.com/pre-commit/pre-commit-hooks
      rev: v4.0.1
      hooks:
        # See https://pre-commit.com/hooks.html for more hooks
        - id: check-ast
        - id: check-case-conflict
        - id: check-executables-have-shebangs
        - id: check-merge-conflict
        - id: debug-statements
        - id: end-of-file-fixer
        - id: name-tests-test
          args: [ "--django" ]
        - id: trailing-whitespace

    - repo: https://github.com/rtts/djhtml
      rev: 'v1.4.10'
      hooks:
        - id: djhtml

    - repo: https://github.com/ambv/black
      rev: 21.12b0
      hooks:
      - id: black

    - repo: local
      hooks:
        - id: django-test
          name: django-test
          entry: python manage.py test -v 1
          always_run: true
          pass_filenames: false
          language: system
```
Run: `pre-commit install`
Finally, run the following command: `pre-commit autoupdate`

<br><br>

<!-- CONTACT -->
## Contact
eduardo@analitika.fr

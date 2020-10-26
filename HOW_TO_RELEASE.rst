Release process
===============
1. the release happens from `master` so make sure it is up-to-date:

   .. code:: sh

      git pull origin master

2. look at `whats-new.rst` and make sure it is complete and with
   references to issues and pull requests

3. open and merge a pull request with these changes and update the release date

4. run the test suite

5. check that the documentation is building

6. Commit the release:

   .. code:: sh

      git commit --allow-empty -am "Release v0.X.Y"

7. Tag the release:

   .. code:: sh

      git tag -a v0.X.Y -m "v0.X.Y"

8. Build source and binary wheels:

   .. code:: sh

      git clean -xdf
      python -m pep517.build --source --binary .

9. Use `twine` to check the package build:

   .. code:: sh

      twine check dist/*

10. try installing the wheel in a new environment and run the tests 

   .. code:: bash

      python -m venv test
      source test/bin/activate
      python -m pip install -r requirements.txt
      python -m pip install pytest
      python -m pip install dist/*.whl
      python -m pytest
      git reset --hard HEAD

11. Push to master and upload to PyPI:

   .. code:: sh

      git push origin master
      git push origin --tags
      twine upload dist/*

   Be careful, this can't be undone.
              
12. Update stable:

    .. code:: sh

       git checkout stable
       git merge v0.X.Y
       git push origin stable

13. Make sure readthedocs builds both `stable` and the new tag

14. Add a new section to `whats-new.rst` and push directly to master
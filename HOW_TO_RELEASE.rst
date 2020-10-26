Release process
===============
1. the release happens from `master` so make sure it is up-to-date:

   .. code:: sh

      git pull origin master

2. look at `whats-new.rst` and make sure it is complete and with
   references to issues and pull requests

3. open and merge a pull request with these changes and the filled in release date

4. run the test suite

5. check that the documentation is building

6. Commit the release:

   .. code:: sh

      git commit --allow-empty -am "Release v0.X.Y"

7. Tag the release and push to master:

   .. code:: sh

      git tag -a v0.X.Y -m "v0.X.Y"
      git push origin --tags

8. Draft a release for the new tag on github. A CI will pick that up, build the project
   and push to PyPI. Be careful, this can't be undone.
              
9. Update stable:

    .. code:: sh

       git checkout stable
       git merge v0.X.Y
       git push origin stable

10. Make sure readthedocs builds both `stable` and the new tag

11. Add a new section to `whats-new.rst` and push directly to master

# To-dos:

*V1 work:*

- [x] Add package version to the top-level scope: `__version__ = version("sawnergy")` in `src/sawnergy/__init__.py`.
- [ ] Document existing API: pre-conditions, post-conditions, specifications.
    Focus on `src/sawnergy/rin/rin_util.py` and `src/sawnergy/sawnergy_util.py` in particular because they will be
    important for v2.
- [ ] Freeze the existing API.
- [ ] The existing API often consumes and produces sawnergy archives. Those must now have attributes specifying
    version of sawnergy used to create them, as well as `schema_name`, that is, whether it is a RIN, walks on a RIN,
    or shallow residue embeddings.
- [ ] Add fail-fast routes for the existing infrastructure when passed incompatible sawnergy archive to.

*V2 work:*

- [ ] Create `v2/` sub-directory of `src/sawnergy/`, therein will be:
    - TBD

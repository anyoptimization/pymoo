# Contributing Guidelines
🎉 Thanks for your interest in helping improve pymoo! 🎉

**If you are looking for pymoo's documentation, go here instead: https://www.pymoo.org/**

These guidelines are for people who want to contribute to pymoo. There are many ways to contribute, such as contributing code, reporting bugs, creating feature requests, helping other users [in our Discussions](https://github.com/anyoptimization/pymoo/discussions) or [in our Discord](https://discord.gg/DUCGrvqWEM), etc.

## Before Contributing
**Before you start coding a bug fix or feature request, please post in the respective GitHub Issue to express your intent to volunteer and wait for a positive response** And if there is no issue created yet, please create one first.

This ensures:
- Multiple developers aren't unknowingly working on the same issue
- The issue is something pymoo's maintainers believe should be implemented
- Any architectural changes that need to be implemented have been sufficiently planned by the community in collaboration with pymoo's maintainers
- Your time is well spent

Please note: Opening a pull request to add a new feature--not just a bug fix or similar--without prior approval from the pymoo team has a very low likelihood of being merged. Introducing new features involves extensive considerations, such as ensuring they meet our standards and support future maintenance.

## Style Guide
We currently do not have a specific style guide for the project, but do follow [PEP-8](https://peps.python.org/pep-0008/) and [PEP-257](https://peps.python.org/pep-0257/)

## Code Standards
Writing good code isn't just about the content; it's also about the style. During Continuous Integration testing, various tools will check your code for stylistic errors, and any warnings will cause the test to fail. Therefore, adhering to good style is essential for submitting code to pymoo.

Additionally, given the widespread use of our library, it's crucial to avoid sudden changes that might break existing user code. We strive to maintain backward compatibility to prevent widespread disruptions.

## Finding an Issue to Contribute To

If you're new to pymoo or open-source development, we suggest exploring the GitHub “Issues” tab to find topics that interest you. Unassigned issues labeled "good first issue" are typically suitable for newer contributors.

After identifying an interesting issue, it’s wise to assign it to yourself to avoid duplicated efforts.

If you can't continue working on the issue, please unassign it so others know it's available again. You can also check the list of assigned issues, as some may not be actively worked on. If you're interested in an assigned issue, kindly ask the current assignee if you can take over.

## Bug Reporting

To help all developers understand the scope of the issue, please be sure to include the following details in your Bug Report issues:

- Summary with a clear and concise description of the problem
- Reproducible Code Example, as self-contained and minimal as possible
- Steps To Reproduce the bug
- Expected Behavior assuming there were no bugs
- Current Behavior of the buggy experience, error messages, stack traces, etc.
- Versioning related to the environment, OS, dependencies, etc. that you are running

## Making a Pull Request
To enhance the likelihood of your pull request being reviewed, you should:

- **Reference an Open Issue:** For significant changes, link to an existing issue to clarify the PR’s purpose.
- **Include Appropriate Tests:** Make sure tests are included; these should be the initial part of any PR.
- **Keep It Simple:** Ensure your pull requests are straightforward; larger PRs require more time to review.
- **Maintain CI Green State:** Ensure continuous integration tests are passing.
- **Regular Updates:** Keep updating your pull request, either upon request or every few days.

## Topics of Interest
Some topics that are, in our opinion, interesting to incorporate in the future:

- **New features:** For instance, new test problems, algorithms, or any other multi-objective related implementation.

- **Constraint Handling:** So far, mostly parameter-less constraint handling is used. Many different strategies of handling constraints have been studied, and some state of the art methods should be provided in pymoo as well.

- **Interactive Visualization:** Our framework provides static visualization for the objective space in higher dimensions. However, it would be nice to make it possible to explore solutions interactively. A suitable choice would be a web-based application with a javascript/typescript based interface using pymoo to answer requests necessary for plotting or optimization

- **Other Topics:** Those are topics that came to our mind. However, there are many more things related to multi-objective optimization that are interesting and standard implementation in a framework would be useful!

If you are interested in any of those topics, please let us know.

## Attribution
These Contributing Guidelines are adapted from [GitHub's Building communities](https://docs.github.com/en/communities/setting-up-your-project-for-healthy-contributions/setting-guidelines-for-repository-contributors), [Contributing to pandas](https://pandas.pydata.org/docs/dev/development/contributing.html), and [Streamlit's Contributing](https://github.com/streamlit/streamlit/wiki/Contributing).
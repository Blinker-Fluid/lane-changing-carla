# lane-changing-carla

## Running

### Starting the CARLA server

You can start the CARLA server by executing: 

    $ sh ./start-server.sh

## How to collaborate

There are two persistent branches: __main__ and __dev__. __main__ is the current working/demo version, and __dev__ is the development branch.

Never directly commit to these. Instead, create a new branch from __dev__ when adding new features.

### Creating new branches

Let's say you are working on a __feature__ called "Implement reward   manager". To make sure we don't have any conflicts, you will need to create a new branch from __dev__ and work on this feature in this branch only. 

The convention we are using for feature branches is as __feat/feature-name__. So, you would name your new branch __feat/implement-reward-manager__.

Here is a handy terminal command for creating new branches:

    $ git branch 
    # make sure you are in the dev branch, then:
    $ git checkout -b feat/feature-name

Substitute feature-name for the task name as written in the task manager. This way, it is clear to everyone what the branch is for.

Other options for branch names include: _fix_, _docs_, _chore_. Check [semantic branch names](https://gist.github.com/seunggabi/87f8c722d35cd07deb3f649d45a31082) for more examples.

#### __DO:__
✅ Fetch dev branch before checking out  
✅ Double check you are on the correct branch  
✅ Use [semantic commit messages](https://gist.github.com/joshbuchea/6f47e86d2510bce28f8e7f42ae84c716)  
✅ Make incremental commits  
✅ Check everything works before committing  

#### __DON'T:__
❌ Modify/add files unrelated to the task at hand  
❌ Make bloated commits  

### Pull requests

After you complete a feature, the next step is to open a pull request to merge the feature branch back to __dev__.
- Use the Pull Requests tab on GitHub to create a new pull request
- Make sure the source and target branches are correct.
- Open the pull request.
- If no conflicts exist, it is probably safe to merge & delete.
- If there are merge conflicts, __do not merge__. Let's talk about it first.
# robosuite

![gallery of_environments](docs/images/gallery.png)

[**[Homepage]**](https://robosuite.ai/) &ensp; [**[White Paper]**](https://arxiv.org/abs/2009.12293) &ensp; [**[Documentations]**](https://robosuite.ai/docs/overview.html) &ensp; [**[ARISE Initiative]**](https://github.com/ARISE-Initiative)

-------

---

Start to work with Linux, Mujoco, Robosuite, and Pycharm
After searching for a guide for creating a working environment on Linux with Mujoco, I combined a few sources, guides and my experience into one guide.
so follow the steps and let's start :)
I'm working with:
- Linux Ubuntu 20.04
- Mujoco 2.0
- Python 3.8
- test

The main steps are:
# A. Download & install Mujoco
# B. Download & install Robosuite
# C. Install python and anaconda
# D. Create a new virtual environment in anaconda
# E. Download & install Mujoco-py
# F. Setup path & environment in PyCharm
# I. Check the environment
note: at the end of each step, there is a "checking" part' to see that the installation and the setup want well. don't skip this part.

---

# A. Download & install Mujoco:
Obtain a 30-day free trial on the MuJoCo website or a free license if you are a student. The license key will arrive in an email with your username and password.
Download the MuJoCo version 2.0 binaries for Linux.
Unzip the downloaded mujoco200 directory into 

$HOME/.mujoco/mujoco200.
 Place your license key (the mjkey.txt file from your email) at 

$HOME/.mujoco/mjkey.txt

# B. Download & install Robosuite:
 Clone the Robosuite repository
$ git clone https://github.com/StanfordVL/robosuite.git
$ cd Robosuite
Install the base requirements with
$ pip3 install -r requirements.txt
if you are going to use OpenAI Gym interfaces, inverse kinematics controllers powered by PyBullet, please install the extra dependencies. 
open the "requirements-extra.txt" file, put in a comment thehidap line (it's only for Mac OS X)
requirements-extra.txt after the change save and run the following terminal:
$ pip3 install -r requirements-extra.txt
Test your installation (don't skip this step):

$ python robosuite/demo.py

# C. Install python and anaconda:
If you have python and anaconda installed on your computer you can skip those steps.
Download python from https://www.python.org/downloads/ and follow the instructions.
Download Anaconda from https://docs.anaconda.com/anaconda/install/linux/ and follow the instructions.

# D. Create a new virtual environment in Anaconda:
Open Anaconda Navigator from the terminal - open the terminal the write: 

$ anaconda-navigator

If it doesn't work. Run this command on your terminal:
$ source ~/anaconda3/bin/activate root
$ anaconda-navigator
In the Environment tab, open a new environment (by pressing the "create" button) and choose the desire python version.

a new environment in anaconda3. after creating the virtual environment, open this environment in the terminal from anaconda:
the terminal will look like that:
in the parenthesis, you will see the environment's name. and then the directory you are in.

# E. Download & install Mujoco-py
Download Mujoco-py from git and install it in the virtual environment that we create (open the terminal as explained in section D3):

$ git clone https://github.com/openai/mujoco-py.git
$ cd mujoco-py
$ python3 setup.py install
2. Test Mujoco-py (don't skip this step):
$ python3
>> import mujoco_py
>> import gym
>> env = gym.make('Hopper-v2')  # or 'Humanoid-v2' 
>> env.render()

# F. Setup path & environment in PyCharm
I'm not sure if it the correct way to approach this issue, but it works for me.
open the Pycharm and enter to 'Edit Configurations' (ender Run in the main tool). 
go to 'Templates', select 'Python', and open the 'Environment variables' by clicking on the key at the end of the command line' on the right.

2. For the Mujoco and Robosuite to word, we need to set two PATH, by pressing +, as follow:
LD_LIBRARY_PATH : /home/user/.mujoco/mujoco200/bin
LD_PRELOAD : /usr/lib/x86_64-linux-gnu/libGLEW.so
the first line you need to change to your home directory.
The second line can copy it- as is.
3. If you want to work with Robosuit, open the Robosuite folder from PyCharm. if you want to word only with Mujoco' open a new project.
4. set the 'Project Interpreter' (enter to Filt → Setting → Project → Project Interpreter) by press the 'setting' button in the upper right corner and then 'add':
add picture
choose 'Conda Environment', and then 'Existing Environment' and set the path of the virtual environment that you create before.
save this setting.

# I. Check the environment
in the Robosuite directory, there is an 'examples' directory. 
open one example and test it.

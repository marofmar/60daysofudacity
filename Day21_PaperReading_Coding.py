'''
OpenAI
<Attacking Machine Learning with Adversarial Examples>
https://openai.com/blog/adversarial-example-research/
'''

# 1. what is Adversarial exmples?
'''
"Adversarial examples are inputs to machine learning models that an attacker has intentially designed 
to cause the model to make a mistake"
- In other words, in my understanding,
 we put the adversarial examples aimimg for higher performance of our model,
and also for the higher security of the system. 
'''

# 2. give me some examples of the adversarial things. I don't follow you YET..
'''
There is an interesting example. In the demonstration from Explaining and Harnessing Adversaril Examples(https://arxiv.org/abs/1412.6572)
From the 57.7% conficdence Panda imgae, if the attacker adds a small perturbation which has 
been calculated beforehand for the purpose of attacking the original one, then the perturbation added
panda image suddenly becomes 99.3% confidence of Gibbon!

'''

# 3. So, why you say this technology can be dangerous?
'''
Let's say you buy an autonomous vehicle. The vehicles are mostly run by the image information 
processing. Then if there are some evil minded persons who wanna attack you, then the person caan add some
stickers or paint to create an adversarial stop sign that the vehicle would interpret as a 'yield' 
or other sign. Reference:Practicala Black-Box Attacks agains Deep Learning Systems using Adversarial Examples (https://arxiv.org/abs/1602.02697)

'''
# 4. Then,, is it possible not the image itself, but the way the ML works, like reinforcement learning can also be affected?
'''
Sure, according to the research from UC Berkeley. OpenAI, and Pennsylvania State University, 
Adversarial Attacks on Neural Network Policies(http://rll.berkeley.edu/adversarial/) and 
research from the Univesity of Nevada at Reno, Vernerability of Deep Reinforcement Learning to Policy Induction Attacks(https://arxiv.org/abs/1701.04143),
the reserches show that widely-used RL algorithms, like DQN, TRPO, A3C aare alal vernerable to adversarial inputs.
The adversarial attack to the RL systems can generate degraded, lower performace even with the very
subtle tiny pertubaations in human's perspective, but causing an RL agent to move opposite from the intended way and so forth. 
Possibly vary dangerous. 

'''
'''
Posting reading 2 
OpenAI
<Attacking Machine Learning with Adversarial Examples>
https://openai.com/blog/adversarial-example-research/

For many parts, I could not get undertand the things easily,
so I will start by picking up the every first sentence of the paragraph, and grasping it afterwards, also trying to make summary of this.

'''

# Attempted defenses against adversarial examples 
'''
Traditional techniques for making amchine learning models more robust, such as weight
decay and dropout, generally do not provide a practical defense against adversarial examples.

Adversarial training: This is a brute force solution where we simply generate a lot of 
adversarial exampeles and explicitly train the model not to be fooled by each of them. 

Defensive distillation: This is a strategy where we train the model to output probabilities
of different classes, rather than hard decisions about which claass to output. 

Yet even these specialized algorithms can easily be broken by giving more computational firepower to the attacker.

'''

# A failed defense: 'gradient masking'
'''
To give an example of how a simple defense can fail, let's conider why a technique called 
'gradient masking' does not work. 

'Gradient masking' is a term introduced in Practicala Black-Box Attacks against Deep Learning Systems using
Adversarial Examples. 

Most adversarial example construction techniques use the gradient of the model to make an attack. 

BUt what if ther ewer no gradient -- what if an infinitesimal modification to the image caused 
no cahnge in the output of the model?

We can easily imagine some very trivial ways to get rid of the gradient. 

Let's run a thought experiment to see how well we could defend our model against adversaril examples 
by running it in 'most likely class' mode instead of 'probability mode.' 

Even more unfortunately, it turns out that the attacker has a very good strategy for guessing 
where the holes in the defense are. 

The defense strategies that perform gradient masking typically result in a model that is 
very smooth in specific directions and neighborhoods of training points, which makes it harder
for the adversary to find gradients indicating good candidate directions to perturb the input in a 
damaging way for the model.

A procedure for performing a model extraction attack was introduced in the black-box attacks paper.

We find that both adversarial training and defensive distillation accidentally perform a kind of
gradient masking.
'''

# Why is it hard to defend agaainst adversarial examples?
'''
Adversarial examples are hard to defend against becasue it is difficult to contruct
a theoretical model of the adversarial example crafting process.

Adversarial examples are also hard to defned against because they require machine learning models 
to produce good outputs for every possible input.

Every strategy we have tested so far fails becuase it is not adaptive: it may block one kind of attack,
but it leaves another vulnerability open to an attacker who knows about the defense being used. 

'''

#Conclusion
'''
Adversarial examples show that many modern machine learning algorithms can be broken in surprising ways. 

'''









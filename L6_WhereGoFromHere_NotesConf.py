'''
Read: Algorithmic Foundation by Cynthia Dwork/ Deep Learning with DP
Topics: The Exp Mechanism/ THe Moment's Accountant/ DP Stochastic Gradient Descent
Advice: For deployment-stick with public framework/ Join the DP community/ DP is till in the early days


Final Project Description

Guest Interview: Ashish Garg, Lead Product Manager-SIRI, AIML, APPLE
Q: How does Apple approach privacy?
A's answer, my understanding summary: Privacy is basic human right, and this cannot be compromised.
Q: What is privacy preserving machine learning?
A's: Introducing privacy to the model, various forms of it, lots of advances in techniques, this area, like in that of federated learning.

Ashish talked about the Face ID of Apple. The surprising fact of this talking was that the data of each user's Face never leaves the device! Wow

Guest Interview: Abhishek Bhowmick, Lead ML Privacy, CORE ML, AIML, APPLE
Q: How does Apple approach privacy?

Differential Privacy is also used in Apple's technology!

THIS IS SOOOOOOOOOO COOOOOOL!!!!!

Guest Interview: Miles Brundage, Research Scientist (Policy part), OpenAI

'''

'''
NVIDIA AI Conference

3. NVIDIA AI Conference in Seoul, South Korea:
	NVIDA Keynote - Marc Hamilton VP/NVIDIA
		- a robot in industry that is programmed with not only ‘if-else’ but interact, smart enough to grasp the surrounding information, and conduct right behaviors based on that.
		- Kaya, Issac
		- Financial industry:
			- Stock trading: previous benchmark 20 CPU 			- run on DGX2: 6000x faster data processing
			- Real-time fraud detection: PayPal, with T4 GPUs
			- Smart City Platform: NVIDIA Metropolis, faster with lower cost
		- NVIDIA is the only company that has full solution for autonomous vehicles
		- NVIDIA Highway Loop: 77 miles, Dec 2018
		- Healthcare: Medical imaging, today 70% of this is conducted with AI, Clara AI Toolkit
	Eunsoo Shim, the head of AI&SW Research Center, Samsung Electronics
		- 삼성종합기술원
		- User Device, Edge Server, Large Scale servers
		-  Recently published CVPR paper
		-  To solve memory bandwidth problem -  >In-memory computing
		- Issues in On-Device Learning: have tendency to loose information previously learned - Elastic weights, algorithms, etc.,
		- [Summary]
			- more and more devices will get on-devide AI capability
			- Strategies in need of AI on device, edge, and cloud
			- Memory bandwidth is a key bottleneck -> algorithm innovation, near-memory/in-memory computing
			- Continuous learning is essential for high level intelligence - > On-device learning without manual labeling is required.

	Getting more DL Training Acceleration using Tensor Cores and AMP
			- AMP: Automatic Mixed Precision - higher model performance with
				 FP16 with Tensor Cores: 8x compute throughput, 2x memory throughput, 1/2x memory storage
				3x faster learning
				PyTorch Apex Package: NVIDIA AMP allows Io implement the three part automatically
				Two components: Automated casting, Automated loss scaling
				Tensorflow AMP
				PyTorch AMP: optimization level from O0 O1 O2 O3: trial and error, O3 only F16, so sometimes learning fails. So out of O1 and O2 try things out (due to the weights things)  (Distributed data parallels should be deployed to run the multi GPU AMP smoothly)
				MxNEt
				HugglingFace’s pertained BERT
				[https://medium.com/tensorflow/automatic-mixed-precision-in-tensorflow-for-faster-ai-training-on-nvidia-gpus-6033234b2540]
				NVIDIA NGC models available
				on demand GTC talks - PyTorch
				Currently working on PyTorch BERT version and will be released as long as it finishes.

	NAVER Clova
			-  이제 40분만 녹음해도 서비스 퀄/ 원래 1년2년전만 해도 10시간 해야 했음
			-  Line Conference yesterday:
			- AutoCut,
	DALI: Fast data pipelines for deep learning
			- for faster image augmentation

	Raj Mirpuri, VP professional visuaa
'''

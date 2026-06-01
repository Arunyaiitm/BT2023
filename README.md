1. Introduction
Brain tumors are one of the most dangerous types of cancer, and finding them early on MRI scans is
critical for treatment. Normally, deep learning models need hundreds or thousands of labelled images to
work. Stember and Shalu (2022) showed that reinforcement learning can localize brain tumors using just
30 training images with a Deep Q-Network (DQN). In my project, I simplified their approach and used
basic tabular Q-learning — a simple NumPy table instead of a neural network, while keeping the same
environment, actions, and reward system from the paper.
2. Dataset
I used the BraTS 2023 Glioma (GLI) dataset containing pre-operative brain MRI scans converted into 2D
axial PNG slices at 240×240 resolution. Each patient has separate folders for MRI modalities (t1c, t1n,
t2f, t2w) and a segmentation mask (seg). I used the T1-contrast (t1c) modality. For each patient, I picked
the axial slice with the largest tumor area, filtered out slices with fewer than 50 tumor pixels, and split
into 30 training and 30 testing images.
3. How It Works
Gridworld: Each 240×240 image is divided into a 4×4 grid of 60×60 pixel cells, creating 16 positions.
The agent starts at the top-left corner (0,0) and navigates to find the tumor.
Actions: The agent has 3 actions — stay still, move down, or move right — matching the paper exactly. It
cannot move up or left.
Rewards (from the paper): Staying still outside the tumor gets −2.0 (strong penalty to force movement).
Moving but still outside the tumor gets −0.5. Moving onto the tumor or staying on the tumor both get
+1.0.
State representation: The state combines the agent's grid position with the brightness level of the current
cell (average pixel intensity binned into 5 categories: background, dim, medium, bright, very bright). This
gives 16 × 5 = 80 states.
Q-Learning: I maintain a Q-table of shape (80, 3) and update it using the Bellman equation: Q(s,a) ←
Q(s,a) + α × [r + γ × max Q(s'
,a') − Q(s,a)], with α=0.1, γ=0.99, and epsilon-greedy exploration starting at
ε=0.7.
4. Results
I trained for 300 episodes (20 steps each) and tested on 30 unseen images. The agent achieved
approximately 60% accuracy — meaning it landed on the tumor in about 18 out of 30 test images.
Method Training Images Test Accuracy
Supervised CNN (paper) 30 ~11%
My Tabular Q-Learning 30 ~60%
DQN (paper) 30 ~70%
5. Problems Faced and Solutions
1. Understanding the paper: The original uses a DQN with CNNs, replay buffers, and target
networks. I simplified to tabular Q-learning, keeping the same gridworld and rewards but
replacing the neural network with a NumPy array.
2. Matching folders: BraTS has a complex folder structure. I wrote a matching function that pairs
t1c and seg folders by case ID and matches slices by their axial number.
3. Selecting good slices: Most slices have no tumor. I scan all segmentation slices per patient and
pick the one with maximum tumor pixels, filtering out weak masks (<50 pixels).
4. Agent walking the same path on every image: This was the biggest issue. My initial state was
position-only (16 states), so the agent was completely blind to the image. It learned one average
policy and followed the same fixed path regardless of where the tumor was. I fixed this by adding
brightness to the state (position × brightness = 80 states), so the agent can distinguish bright cells
(potential tumor) from dark ones.
5. Brightness bins too coarse: My first attempt used 2 bins (dark/bright, threshold 40). But all brain
tissue has intensity above 40, so every image produced the same brightness map. I printed actual
intensity values and discovered tumor cells hit 200+ while normal brain is 70–130. Switching to 5
bins with thresholds at 10, 50, 100, 140 fixed this — the agent started taking different paths on
different images.
6. Same result every run: I had random.seed(42) set, making every run deterministic. Removing the
seed confirmed that parameter changes actually produce different results; I kept it for final
reproducibility.
6. Limitations
The agent can only move down and right — if it overshoots the tumor, it cannot go back. This 3-action
constraint is acknowledged by the paper's authors as a known limitation. The 4×4 grid gives only coarse
localization (60×60 blocks, not pixel-level). I also only analyze 2D slices, not full 3D volumes.
7. Conclusion
I implemented a simplified tabular Q-learning agent for brain tumor localization on BraTS 2023 MRI
images. The main challenge was designing a state that lets the agent actually see the image —
position-only states made it blind, and adding brightness bins solved this. My agent achieves ~60%
accuracy with just 30 training images and no neural network, confirming the paper's core finding that
reinforcement learning is data-efficient for medical image analysis.

# Brain Tumor Localization using Tabular Q-Learning

This project implements a simplified reinforcement learning approach for localizing brain tumors in MRI scans. It is inspired by the work of Stember and Shalu (2022), who showed that reinforcement learning can localize tumors using only a small number of labelled medical images.

Instead of using a Deep Q-Network (DQN), this project uses **basic tabular Q-learning** with a NumPy-based Q-table. The goal was to keep the environment, actions, and reward system similar to the original paper, while making the implementation simpler, easier to understand, and more beginner-friendly.

## Project Overview

Brain tumor localization is an important task in medical image analysis. Traditional deep learning models usually require large labelled datasets, but medical datasets are often limited and difficult to annotate.

This project explores whether a simpler reinforcement learning agent can learn to localize tumors using only a small number of MRI images. The agent moves inside a grid created from the MRI image and learns to reach the grid cell containing the tumor.

## Dataset

The project uses the **BraTS 2023 Glioma (GLI)** dataset.

The original MRI scans were converted into 2D axial PNG slices of size **240 × 240** pixels. Each patient folder contains different MRI modalities and a segmentation mask.

The modality used in this project is:

* `t1c` — T1-contrast MRI image

The segmentation mask used is:

* `seg` — tumor mask

For each patient, the axial slice with the largest tumor area was selected. Slices with fewer than 50 tumor pixels were ignored. The final split used:

* **30 training images**
* **30 testing images**

## Methodology

Each MRI image is divided into a **4 × 4 grid**.

Since each image is 240 × 240 pixels, each grid cell has size:

```text
60 × 60 pixels
```

This gives a total of **16 possible positions** for the agent.

The agent always starts from the top-left corner of the image:

```text
(0, 0)
```

The goal is to move through the grid and reach the tumor-containing cell.

## Actions

The agent has only three possible actions:

```text
0 = stay still
1 = move down
2 = move right
```

The agent cannot move up or left. This matches the action constraint used in the original paper.

## Reward System

The reward system is based on the paper:

| Situation                          | Reward |
| ---------------------------------- | -----: |
| Staying still outside the tumor    |   -2.0 |
| Moving but still outside the tumor |   -0.5 |
| Moving onto the tumor              |   +1.0 |
| Staying on the tumor               |   +1.0 |

The strong penalty for staying still outside the tumor forces the agent to explore and move through the image.

## State Representation

The first version of the project used only the agent’s grid position as the state. This created only 16 states.

However, this made the agent almost blind to the image. It learned one fixed path and followed the same route for every MRI scan, regardless of where the tumor was located.

To fix this, brightness information was added to the state.

Each cell’s average pixel intensity is calculated and placed into one of five brightness categories:

```text
0 = background
1 = dim
2 = medium
3 = bright
4 = very bright
```

So the final state is:

```text
state = grid position × brightness level
```

This gives:

```text
16 positions × 5 brightness levels = 80 states
```

The Q-table therefore has the shape:

```text
(80, 3)
```

where 80 is the number of states and 3 is the number of actions.

## Q-Learning

The agent is trained using the standard Q-learning update rule:

```text
Q(s, a) ← Q(s, a) + α [r + γ max Q(s', a') − Q(s, a)]
```

The parameters used are:

| Parameter         | Value |
| ----------------- | ----: |
| Learning rate α   |   0.1 |
| Discount factor γ |  0.99 |
| Initial epsilon ε |   0.7 |
| Training episodes |   300 |
| Steps per episode |    20 |

The agent uses an epsilon-greedy strategy during training to balance exploration and exploitation.

## Results

The model was trained on 30 MRI images and tested on 30 unseen MRI images.

The tabular Q-learning agent achieved approximately:

```text
60% test accuracy
```

This means the agent successfully landed on the tumor-containing grid cell in around **18 out of 30 test images**.

## Comparison

| Method                           | Training Images | Test Accuracy |
| -------------------------------- | --------------: | ------------: |
| Supervised CNN from paper        |              30 |          ~11% |
| This project: Tabular Q-Learning |              30 |          ~60% |
| DQN from paper                   |              30 |          ~70% |

The result shows that even a simple tabular Q-learning agent can perform reasonably well with very limited training data.

## Key Challenges and Solutions

### 1. Understanding the Original Paper

The original paper used a Deep Q-Network with CNNs, replay buffers, and target networks. To make the project easier to understand and implement, I replaced the neural network with a simple NumPy Q-table while keeping the same gridworld idea, actions, and reward system.

### 2. Matching MRI and Mask Folders

The BraTS dataset has a complex folder structure. I wrote a matching function to pair the `t1c` image folders with their corresponding `seg` mask folders using the case ID and axial slice number.

### 3. Selecting Useful Slices

Most MRI slices do not contain visible tumors. To solve this, I scanned all segmentation masks for each patient and selected the slice with the largest tumor area. Very weak masks with fewer than 50 tumor pixels were removed.

### 4. Agent Following the Same Path

The first version used only the agent’s position as the state. This caused the agent to follow the same path for every image because it had no image-based information.

This was fixed by adding brightness information to the state. After this change, the agent could distinguish darker and brighter cells and take different paths for different images.

### 5. Choosing Better Brightness Bins

The first brightness method used only two bins: dark and bright. This did not work well because most brain tissue had similar intensity values.

After checking the actual intensity values, I found that tumor regions often had much higher brightness values. I changed the state representation to use five brightness bins, which helped the agent make better decisions.

### 6. Reproducibility

At first, the use of a fixed random seed made every run produce the same result. Removing the seed helped confirm that parameter changes affected the results. For the final version, the seed was kept to make the results reproducible.

## Limitations

This project has some limitations:

* The agent can only move down or right.
* If the agent moves past the tumor, it cannot go back.
* The 4 × 4 grid gives only coarse localization.
* Each predicted tumor region is a 60 × 60 block, not a pixel-level tumor boundary.
* The project uses 2D MRI slices, not full 3D MRI volumes.
* The model uses brightness-based states, which are simpler than deep image features.

## Conclusion

This project shows that a simple tabular Q-learning agent can localize brain tumors on MRI images using a very small training set. The most important improvement was adding brightness information to the state representation. Without it, the agent was blind to the image and followed the same path every time.

With only 30 training images and no neural network, the agent achieved around 60% accuracy on unseen test images. This supports the idea that reinforcement learning can be useful for data-efficient medical image analysis.

## Technologies Used

* Python
* NumPy
* OpenCV
* Matplotlib
* Reinforcement Learning
* Q-Learning
* Medical Image Processing
* BraTS 2023 Dataset

## Reference

Stember, J. N., & Shalu, H. (2022). Reinforcement learning for brain tumor localization in medical images.



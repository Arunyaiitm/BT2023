# IMPORTING 
import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import pickle

print('All imports OK')


#CONFIGURATION
TRAIN_DIR = '/Users/arunyahooda/Desktop/BT_QLearn/Training_png'
RESULTS_DIR = '/Users/arunyahooda/Desktop/BT_QLearn/results'


IMG_SIZE    = 240      # images are 240x240
GRID_SIZE   = 4        # 4x4 grid → each cell is 60x60 pixels
CELL_SIZE   = IMG_SIZE // GRID_SIZE   # 60
N_BRIGHTNESS = 5
N_STATES     = GRID_SIZE * GRID_SIZE * N_BRIGHTNESS   # 80 states
N_ACTIONS   = 3        # 0=stay, 1=move down, 2=move right

N_TRAIN     = 30      # number of training images (same as paper)
N_TEST      = 30       # number of testing images
N_EPISODES  = 300       # training episodes per image
N_STEPS     = 20       # steps per episode

ALPHA       = 0.1      # learning rate
GAMMA       = 0.99     # discount factor
EPSILON     = 0.7      # initial exploration rate
EPSILON_MIN = 1e-4
EPSILON_DECAY = 1e-4   # decay per episode

os.makedirs(RESULTS_DIR, exist_ok=True)
print(f'Cell size: {CELL_SIZE}px | States: {N_STATES} | Actions: {N_ACTIONS}')


# LOAD IMAGE PATH
# LOAD IMAGE PATH
def get_slice_number(filename):
    """Get the axial slice number from a filename."""
    name = os.path.basename(filename)
    part = name.split('axial_')[-1]
    number = part.replace('.png', '')
    return number


def count_tumor_pixels(seg_path):
    """Count non-zero pixels in one segmentation mask."""
    seg = np.array(Image.open(seg_path).convert('L'))
    return np.sum(seg > 0)


def get_image_pairs(root_dir):
    """
    Scans root_dir for matching t1c + seg folder pairs.
    For each case, it checks all segmentation slices and picks the slice
    with the biggest tumor area.
    """
    pairs = []
    all_folders = sorted(os.listdir(root_dir))

    case_ids = set()
    for f in all_folders:
        if f.endswith('-t1c'):
            case_ids.add(f.replace('-t1c', ''))

    for case_id in sorted(case_ids):
        t1c_folder = os.path.join(root_dir, case_id + '-t1c', 'vol', 'images')
        seg_folder = os.path.join(root_dir, case_id + '-seg', 'vol', 'images')

        if not os.path.exists(t1c_folder) or not os.path.exists(seg_folder):
            continue

        t1c_imgs = sorted([f for f in os.listdir(t1c_folder) if f.endswith('.png')])
        seg_imgs = sorted([f for f in os.listdir(seg_folder) if f.endswith('.png')])

        if not t1c_imgs or not seg_imgs:
            continue

        t1c_dict = {}
        for f in t1c_imgs:
            t1c_dict[get_slice_number(f)] = f

        best_tumor_pixels = 0
        best_t1c_path = None
        best_seg_path = None

        for seg_file in seg_imgs:
            slice_no = get_slice_number(seg_file)

            if slice_no not in t1c_dict:
                continue

            seg_path = os.path.join(seg_folder, seg_file)
            tumor_pixels = count_tumor_pixels(seg_path)

            if tumor_pixels > best_tumor_pixels:
                best_tumor_pixels = tumor_pixels
                best_seg_path = seg_path
                best_t1c_path = os.path.join(t1c_folder, t1c_dict[slice_no])

        if best_t1c_path is not None and best_seg_path is not None:
            pairs.append((best_t1c_path, best_seg_path))
            print(case_id, '| best tumor pixels:', best_tumor_pixels)

    return pairs
all_pairs = get_image_pairs(TRAIN_DIR)


#FILTER
def has_tumor(seg_path, threshold=50):
    """
    Returns True if the segmentation mask has enough non-zero pixels.
    threshold = minimum number of tumor pixels to count as a valid slice.
    """
    seg = np.array(Image.open(seg_path).convert('L'))
    return np.sum(seg > 0) >= threshold


valid_pairs = [p for p in all_pairs if has_tumor(p[1])]
print(f'Pairs with visible tumor: {len(valid_pairs)} / {len(all_pairs)}')


# Shuffle and split into train/test
random.seed(42)
random.shuffle(valid_pairs)


train_pairs = valid_pairs[:N_TRAIN]
test_pairs  = valid_pairs[N_TRAIN : N_TRAIN + N_TEST]

print(f'Train: {len(train_pairs)} | Test: {len(test_pairs)}')


#VISUALIZE
def show_sample(t1c_path, seg_path):
    mri = np.array(Image.open(t1c_path).convert('L'))
    seg = np.array(Image.open(seg_path).convert('L'))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(mri, cmap='gray');  axes[0].set_title('MRI (t1c)'); axes[0].axis('off')
    axes[1].imshow(seg, cmap='hot');   axes[1].set_title('Mask (seg)'); axes[1].axis('off')

    # Overlay
    overlay = np.stack([mri]*3, axis=-1).astype(np.uint8)
    overlay[seg > 0] = [255, 0, 0]  # red tumor
    axes[2].imshow(overlay);          axes[2].set_title('Overlay'); axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'sample_check.png'), dpi=100)
    plt.show()

show_sample(*train_pairs[0])


#GRID
def get_brightness(row, col, mri_array):
    r1 = row * CELL_SIZE
    c1 = col * CELL_SIZE
    patch = mri_array[r1:r1+CELL_SIZE, c1:c1+CELL_SIZE]
    avg = np.mean(patch)

    if avg < 10:
        return 0    # background (black)
    elif avg < 50:
        return 1    # dim
    elif avg < 100:
        return 2    # medium
    elif avg < 160:
        return 3    # bright
    else:
        return 4    # very bright (likely tumor)


def pos_to_state(row, col, mri_array):
    """State = position (0-15) * 2 + brightness (0 or 1) = 0..31"""
    b = get_brightness(row, col, mri_array)
    return (row * GRID_SIZE + col) * N_BRIGHTNESS + b


def state_to_pos(state):
    """Convert state index back to (row, col, brightness)."""
    pos = state // N_BRIGHTNESS
    b = state % N_BRIGHTNESS
    return pos // GRID_SIZE, pos % GRID_SIZE, b


def take_action(row, col, action):
    """
    Actions: 0=stay, 1=move down, 2=move right
    Agent stays at grid boundary (does not wrap).
    """
    if action == 1:  # down
        row = min(row + 1, GRID_SIZE - 1)
    elif action == 2:  # right
        col = min(col + 1, GRID_SIZE - 1)
    return row, col


def agent_overlaps_tumor(row, col, seg_array):
    """
    Check if the 60x60 block at (row, col) has any tumor pixels.
    """
    r1 = row * CELL_SIZE
    c1 = col * CELL_SIZE
    patch = seg_array[r1:r1+CELL_SIZE, c1:c1+CELL_SIZE]
    return np.any(patch > 0)


def get_reward(row, col, new_row, new_col, action, seg_array):
    """
    Reward scheme from the paper (Fig 1):
      -2  : stayed still, not on tumor
      +1  : stayed still, on tumor
      -0.5: moved, ended outside tumor
      +1  : moved, ended on tumor
    """
    on_tumor = agent_overlaps_tumor(new_row, new_col, seg_array)
    stayed   = (action == 0)

    if stayed and not on_tumor:
        return -2.0
    elif stayed and on_tumor:
        return +1.0
    elif not stayed and not on_tumor:
        return -0.5
    else:  # moved and on tumor
        return +1.0


print('Grid helpers defined — states:', N_STATES, '| actions:', N_ACTIONS)


#Q-LEARNING
# Q-table: shape (N_STATES, N_ACTIONS) — one table shared across all training images
Q = np.zeros((N_STATES, N_ACTIONS))

epsilon = EPSILON
episode_rewards = []   # track average reward per episode

for episode in range(N_EPISODES):

    t1c_path, seg_path = random.choice(train_pairs)
    seg_array = np.array(Image.open(seg_path).convert('L'))
    mri_array = np.array(Image.open(t1c_path).convert('L'))  # ADD THIS

    row, col = 0, 0
    state = pos_to_state(row, col, mri_array)  # ADD mri_array
    total_reward = 0

    for step in range(N_STEPS):

        if random.random() < epsilon:
            action = random.randint(0, N_ACTIONS - 1)
        else:
            action = np.argmax(Q[state])

        new_row, new_col = take_action(row, col, action)
        new_state = pos_to_state(new_row, new_col, mri_array)  # ADD mri_array

        reward = get_reward(row, col, new_row, new_col, action, seg_array)
        total_reward += reward

        best_next = np.max(Q[new_state])
        Q[state, action] += ALPHA * (reward + GAMMA * best_next - Q[state, action])

        row, col = new_row, new_col
        state = new_state

    episode_rewards.append(total_reward)

    # Decay epsilon
    epsilon = max(EPSILON_MIN, epsilon - EPSILON_DECAY)

    if (episode + 1) % 10 == 0:
        print(f'Episode {episode+1:3d}/{N_EPISODES} | Avg Reward: {np.mean(episode_rewards[-10:]):.2f} | epsilon: {epsilon:.4f}')

print('\nTraining complete!')

# Save Q-table
np.save(os.path.join(RESULTS_DIR, 'q_table.npy'), Q)
print('Q-table saved.')


#PLOT
plt.figure(figsize=(10, 4))
plt.plot(episode_rewards, alpha=0.4, color='steelblue', label='Episode reward')

# Smoothed line
window = 10
smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
plt.plot(range(window-1, len(episode_rewards)), smoothed, color='red', linewidth=2, label=f'{window}-ep moving avg')

plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Q-Learning Training Rewards')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'training_rewards.png'), dpi=150)
plt.show()
print('Saved training_rewards.png')


#TESTING
def test_agent(Q, t1c_path, seg_path, n_steps=N_STEPS):
    seg_array = np.array(Image.open(seg_path).convert('L'))
    mri_array = np.array(Image.open(t1c_path).convert('L'))  # ADD THIS
    row, col  = 0, 0
    path = [(row, col)]

    for _ in range(n_steps):
        state  = pos_to_state(row, col, mri_array)  # ADD mri_array
        action = np.argmax(Q[state])
        row, col = take_action(row, col, action)
        path.append((row, col))

    success = agent_overlaps_tumor(row, col, seg_array)
    return success, path, seg_array


# Run on all test images
results = []
for t1c_path, seg_path in test_pairs:
    success, path, seg_array = test_agent(Q, t1c_path, seg_path)
    results.append(success)

tp = sum(results)
fp = len(results) - tp
accuracy = tp / len(results) * 100

print(f'Test images  : {len(results)}')
print(f'True Positive: {tp}')
print(f'False Positive: {fp}')
print(f'Accuracy     : {accuracy:.1f}%')


#VISUALIZE AGENT PATH
def visualize_agent(t1c_path, seg_path, Q, save_name='agent_result.png'):
    mri = np.array(Image.open(t1c_path).convert('L'))
    seg = np.array(Image.open(seg_path).convert('L'))

    success, path, _ = test_agent(Q, t1c_path, seg_path)
    final_row, final_col = path[-1]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, img, title in zip(axes, [mri, mri], ['Agent Path', 'Final Position']):
        ax.imshow(img, cmap='gray')
        # Draw grid
        for i in range(1, GRID_SIZE):
            ax.axhline(i * CELL_SIZE, color='lime', linewidth=0.8, alpha=0.6)
            ax.axvline(i * CELL_SIZE, color='lime', linewidth=0.8, alpha=0.6)
        # Overlay tumor in red
        ax.imshow(seg, cmap='Reds', alpha=0.4)
        ax.set_title(title)
        ax.axis('off')

    # Draw agent path on left plot
    for r, c in path:
        cx = c * CELL_SIZE + CELL_SIZE // 2
        cy = r * CELL_SIZE + CELL_SIZE // 2
        axes[0].plot(cx, cy, 'bo', markersize=6)
    # Connect path with line
    xs = [c * CELL_SIZE + CELL_SIZE//2 for r, c in path]
    ys = [r * CELL_SIZE + CELL_SIZE//2 for r, c in path]
    axes[0].plot(xs, ys, 'b-', linewidth=1.5, alpha=0.7)

    # Draw final agent block on right plot
    r1 = final_row * CELL_SIZE
    c1 = final_col * CELL_SIZE
    color = 'lime' if success else 'red'
    rect = plt.Rectangle((c1, r1), CELL_SIZE, CELL_SIZE,
                          linewidth=3, edgecolor=color, facecolor=color, alpha=0.3)
    axes[1].add_patch(rect)
    result_text = 'HIT ✓' if success else 'MISS ✗'
    axes[1].set_title(f'Final Position — {result_text}', color=color)

    plt.suptitle(os.path.basename(t1c_path), fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, save_name), dpi=150)
    plt.show()


# Show first 3 test cases
for i, (t1c, seg) in enumerate(test_pairs[:3]):
    visualize_agent(t1c, seg, Q, save_name=f'test_case_{i+1}.png')


#HEATMAP
def visualize_agent(t1c_path, seg_path, Q, save_name='agent_result.png'):
    mri = np.array(Image.open(t1c_path).convert('L'))
    seg = np.array(Image.open(seg_path).convert('L'))

    success, path, _ = test_agent(Q, t1c_path, seg_path)
    final_row, final_col = path[-1]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    for ax, img, title in zip(axes, [mri, mri], ['Agent Path', 'Final Position']):
        ax.imshow(img, cmap='gray')
        # Draw grid
        for i in range(1, GRID_SIZE):
            ax.axhline(i * CELL_SIZE, color='lime', linewidth=0.8, alpha=0.6)
            ax.axvline(i * CELL_SIZE, color='lime', linewidth=0.8, alpha=0.6)
        # Overlay tumor in red
        ax.imshow(seg, cmap='Reds', alpha=0.4)
        ax.set_title(title)
        ax.axis('off')

    # Draw agent path on left plot
    for r, c in path:
        cx = c * CELL_SIZE + CELL_SIZE // 2
        cy = r * CELL_SIZE + CELL_SIZE // 2
        axes[0].plot(cx, cy, 'bo', markersize=6)
    # Connect path with line
    xs = [c * CELL_SIZE + CELL_SIZE//2 for r, c in path]
    ys = [r * CELL_SIZE + CELL_SIZE//2 for r, c in path]
    axes[0].plot(xs, ys, 'b-', linewidth=1.5, alpha=0.7)

    # Draw final agent block on right plot
    r1 = final_row * CELL_SIZE
    c1 = final_col * CELL_SIZE
    color = 'lime' if success else 'red'
    rect = plt.Rectangle((c1, r1), CELL_SIZE, CELL_SIZE,
                          linewidth=3, edgecolor=color, facecolor=color, alpha=0.3)
    axes[1].add_patch(rect)
    result_text = 'HIT ✓' if success else 'MISS ✗'
    axes[1].set_title(f'Final Position — {result_text}', color=color)

    plt.suptitle(os.path.basename(t1c_path), fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, save_name), dpi=150)
    plt.show()


# Show first 3 test cases
for i, (t1c, seg) in enumerate(test_pairs[:3]):
    visualize_agent(t1c, seg, Q, save_name=f'test_case_{i+1}.png')

#FINAL SUMMARY
print(f'Training images: {N_TRAIN}')
print(f'Testing images: {N_TEST}')
print(f'Episodes: {N_EPISODES}')
print(f'Grid: {GRID_SIZE}x{GRID_SIZE}')
print(f'Steps per episode: {N_STEPS}')
print(f'True Positives: {tp} / {len(results)}')
print(f'Accuracy: {accuracy:.1f}%')
print(f'Results saved to: {RESULTS_DIR}')



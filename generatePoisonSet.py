import os, random, pickle
import torch
from torchvision import datasets, transforms
from tqdm import tqdm

DATA_ROOT = './phantom_data'
os.makedirs(DATA_ROOT, exist_ok=True)

# --- Attack parameters ---
PDR = 0.3     # fraction of training samples to poison
PV  = 1     # pattern visibility

# --- Load CIFAR-10 ---
transform = transforms.ToTensor()
train_set = datasets.CIFAR10(DATA_ROOT, train=True,  download=True, transform=transform)
test_set  = datasets.CIFAR10(DATA_ROOT, train=False, download=True, transform=transform)

# --- Build a small "guessed labeled" pool X_A from true labels ---
num_labels = 250
all_indices = list(range(len(train_set)))
random.shuffle(all_indices)
X_A_indices = all_indices[:num_labels]
X_A = [train_set[i][0] for i in X_A_indices]  # only need tensors, labels unused for poison

# --- crop and concatenate two patterns ---
def make_pattern(p1, p2):
    # assume p1,p2 are [C,H,W] and same size as originals
    C, H, W = p1.shape
    half = W // 2
    # left half from p1, right half from p2
    left  = p1[:, :, :half]
    right = p2[:, :, half:]
    return torch.cat([left, right], dim=2)  # [C,H,W]

# --- Poisoned dataset creation ---
poisoned_indices = set(random.sample(range(len(train_set)), int(len(train_set)*PDR)))
poisoned_data = []

for idx in tqdm(range(len(train_set)), desc="Crafting Phantom poisons"):
    img, lbl = train_set[idx]
    if idx in poisoned_indices:
        # pick two random patterns from X_A (ideally different classes in reality)
        p1, p2 = random.sample(X_A, 2)
        pattern = make_pattern(p1, p2)
        # blend original and pattern per eq. (2)
        img = (1 - PV)*img + PV*pattern
    poisoned_data.append((img, lbl))

# --- Save out as pikles ---
with open(os.path.join(DATA_ROOT, 'phantom_train.pkl'), 'wb') as f:
    pickle.dump({
        'images': torch.stack([x for x,_ in poisoned_data]),
        'labels': torch.tensor([y for _,y in poisoned_data])
    }, f)

with open(os.path.join(DATA_ROOT, 'test.pkl'), 'wb') as f:
    pickle.dump({
        'images': torch.stack([x for x,_ in test_set]),
        'labels': torch.tensor([y for _,y in test_set])
    }, f)



#Visualize poison set
import pickle
import matplotlib.pyplot as plt
import random

# 1) Load the poisoned training file you generated
with open('phantom_data/phantom_train.pkl', 'rb') as f:
    data = pickle.load(f)
poisoned_imgs = data['images']      # Tensor shape [N,3,H,W]
labels         = data['labels']     # Tensor shape [N]

# 2) Reload the original CIFAR-10 train set for comparison
transform = transforms.ToTensor()
orig = datasets.CIFAR10('phantom_data', train=True, download=True, transform=transform)
orig_imgs = torch.stack([orig[i][0] for i in range(len(orig))])

class_names = orig.classes  # ['airplane','automobile',...,'truck']

# 3) Pick a few random samples to visualize
num_examples = 10
indices = random.sample(range(len(orig_imgs)), num_examples)

resize_transform = transforms.Resize((256, 256))

# 4) Plot 
fig, axes = plt.subplots(num_examples, 2, figsize=(6, 3*num_examples))

for row, idx in enumerate(indices):
    clean = orig_imgs[idx]
    bad   = poisoned_imgs[idx]

    clean_resized = resize_transform(clean)
    bad_resized = resize_transform(bad)



    # Clean
    axes[row,0].imshow(clean_resized.permute(1,2,0))
    # axes[row,0].set_title(f"Original, label={labels[idx].item()}")
    axes[row,0].set_title(f"Original: {class_names[labels[idx].item()]}")
    axes[row,0].axis('off')

    # Poisoned
    axes[row,1].imshow(bad_resized.permute(1,2,0))
    axes[row,1].set_title("Poisoned")
    axes[row,1].axis('off')

plt.tight_layout()
plt.show()


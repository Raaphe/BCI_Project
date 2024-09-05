import os
import pathlib
import torch
import random;
import matplotlib.pyplot as plt;


from torch import nn;
from torchvision import datasets;
from PIL import Image
from torch.utils.data import DataLoader;
from pathlib import Path;
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple, Dict, List

DATA_PATH = Path("data/");
IMAGE_PATH = DATA_PATH / "pizza_steak_sushi";
DEVICE = "cuda" if torch.cuda.is_available() else "cpu";

train_dir = IMAGE_PATH / "train";
test_dir = IMAGE_PATH / "test";

class FashionMNISTModel2(nn.Module):
    """
    Model architecture that replicates the TinyVGG
    model from cnn explainer website.
    """
    def __init__(self, input_shape:int, hidden_units:int, output_shape:int, device: torch.device = DEVICE):
        super().__init__();

        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape, 
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units, 
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        );
    
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units, 
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=hidden_units, 
                out_channels=hidden_units,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        );

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*16*16, # we first have the input shape of the image which is 64*64 which then gets maxpooled to 32*32 then to 16*16. The tensor shape that we get from conv_block_2 is that of 1*10*7*7..
                    out_features=output_shape)
        );


    def forward(self, x: torch.Tensor): 
        return self.classifier(self.conv_block_2(self.conv_block_1(x)))

def plot_transformed_images(image_paths, transform, n=3, seed=42):
    """Plots a series of random images from image_paths.

    Will open n image paths from image_paths, transform them
    with transform and plot them side by side.

    Args:
        image_paths (list): List of target image paths. 
        transform (PyTorch Transforms): Transforms to apply to images.
        n (int, optional): Number of images to plot. Defaults to 3.
        seed (int, optional): Random seed for the random generator. Defaults to 42.
    """
    random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f) 
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")

            # Transform and plot image
            # Note: permute() will change shape of image to suit matplotlib 
            # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
            transformed_image = transform(f).permute(1, 2, 0) 
            ax[1].imshow(transformed_image) 
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)


# 1. Take in a Dataset as well as a list of class names
def display_random_images(dataset: torch.utils.data.dataset.Dataset,
                          classes: List[str] = None,
                          n: int = 10,
                          display_shape: bool = True,
                          seed: int = None):
    
    # 2. Adjust display if n too high
    if n > 10:
        n = 10
        display_shape = False
        print(f"For display purposes, n shouldn't be larger than 10, setting to 10 and removing shape display.")
    
    # 3. Set random seed
    if seed:
        random.seed(seed)

    # 4. Get random sample indexes
    random_samples_idx = random.sample(range(len(dataset)), k=n)

    # 5. Setup plot
    plt.figure(figsize=(16, 8))

    # 6. Loop through samples and display random samples 
    for i, targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = dataset[targ_sample][0], dataset[targ_sample][1]

        # 7. Adjust image tensor shape for plotting: [color_channels, height, width] -> [color_channels, height, width]
        targ_image_adjust = targ_image.permute(1, 2, 0)

        # Plot adjusted samples
        plt.subplot(1, n, i+1)
        plt.imshow(targ_image_adjust)
        plt.rcParams['font.size'] = 5
        plt.axis("off")
        if classes:
            title = f"class: {classes[targ_label]}"
            if display_shape:
                title = title + f"\nshape: {targ_image_adjust.shape}"
        plt.title(title)

# 1. subclass Dataset
class ImageFolderCustom(Dataset):

    # 2.Initialize with a targ_dir and transform (optional) parameter
    def __init__(self, targ_dir: str, transform=None) -> None:
        # 3. Create class attributes
        # Get all image paths
        self.paths = list(pathlib.Path(targ_dir).glob("*/*.jpg")) # note: you'd have to update this if you've got .png's or .jpeg's
        # setup transforms
        self.transform = transform;
        # create classes and class_to_idx attributes
        self.classes, self.class_to_idx = find_classes(targ_dir);

    # 4. Make function to load images
    def load_image(self, index:int) -> Image.Image:
        """Opens an image via a path and returns it."""
        image_path = self.paths[index];
        return Image.open(image_path);

    # 5. overwrite the __len__() method (optional but recommended for subclasses of torch.utils.data.Datasets).
    def __len__(self) -> int:
        '''Returns the total number of samples.'''
        return len(self.paths)

    # Overwrite the __getitem__() method (required for subclasses of torch.utils.data.Datasets).
    def __getitem__(self, index:int) -> Tuple[torch.Tensor, int]:
        """Returns one sample of data, (X, y)."""
        img = self.load_image(index=index);
        class_name = self.paths[index].parent.name # expects path in data_folder/class_name/image.jpeg
        class_idx = self.class_to_idx[class_name];

        # Transform if necessary 
        if self.transform:
            return self.transform(img), class_idx 
        else:
            return img, class_idx;

# Make function to find classes in target directory
def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folder names in a target directory.
    
    Assumes target directory is in standard image classification format.

    Args:
        directory (str): target directory to load classnames from.

    Returns:
        Tuple[List[str], Dict[str, int]]: (list_of_class_names, dict(class_name: idx...))
    
    Example:
        find_classes("food_images/train")
        >>> (["class_1", "class_2"], {"class_1": 0, ...})
    """
    # 1. Get the class names by scanning the target directory
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    
    # 2. Raise an error if class names not found
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
        
    # 3. Create a dictionary of index labels (computers prefer numerical rather than string labels)
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

def main():
    target_directory = train_dir;

    class_names_found = find_classes(train_dir);
    print(f"class names found {class_names_found}");

    # Augment training data.
    train_transforms = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.RandomHorizontalFlip(),
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.ToTensor()
    ]);

    # Augment testing data
    test_transforms = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor()
    ]);


    train_data_custom = ImageFolderCustom(
        targ_dir=train_dir, 
        transform=train_transforms
    );

    test_data_custom = ImageFolderCustom(
        targ_dir=test_dir, 
        transform=test_transforms
    );

    print(len(train_data_custom.classes));
    print(len(test_data_custom.class_to_idx));

        # Display random images from ImageFolder created Dataset
    display_random_images(train_data_custom, 
                        n=5, 
                        classes=train_data_custom.classes,
                        seed=None)


    train_dataset = DataLoader(
        batch_size=1,
        dataset=train_data_custom,
        shuffle=True,
        num_workers=0
    );        

    test_dataset = DataLoader(
        batch_size=1,
        dataset=test_data_custom,
        shuffle=False,
        num_workers=0
    );        

    plot_transformed_images(image_paths=list(IMAGE_PATH.glob("*/*/*.jpg")), n=3, seed=None,transform=train_transforms);
    
    img, label = next(iter(train_dataset));

    print(f"image = {img.shape}");
    print(f"image = {label.shape}");
    
    # Creating transforms and loading data into model 0
    simple_transforms = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor()
    ]);

    train_data_simple = datasets.ImageFolder(root=train_dir, transform=simple_transforms);
    test_data_simple = datasets.ImageFolder(root=test_dir, transform=simple_transforms);

    NUM_WORKERS = os.cpu_count();
    BATCH_SIZE = 32;

    train_dataloader = DataLoader(
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        dataset=train_data_simple, 
        num_workers=NUM_WORKERS,
    );

    train_dataloader = DataLoader(
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        dataset=train_data_simple, 
        num_workers=NUM_WORKERS,
    );

    model = FashionMNISTModel2(
        device=DEVICE,
        output_shape=len(class_names_found),
        input_shape=3,
        hidden_units=30
    ).to(DEVICE);

    print(model);
    print(model.state_dict());

    
    
    pass;

if __name__ == "__main__":
    main();
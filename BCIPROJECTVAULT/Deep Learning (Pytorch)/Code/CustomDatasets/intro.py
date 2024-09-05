import torch;
import os;
import torch.nn;
import requests
import zipfile
import matplotlib.pyplot as plt;
import os
import random
from pathlib import Path
from torchvision import transforms;
from torchvision import datasets;
from PIL import Image
from torch.utils.data import DataLoader

DATA_PATH = Path("data/");
IMAGE_PATH = DATA_PATH / "pizza_steak_sushi";
DEVICE = "cuda" if torch.cuda.is_available() else "cpu";

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
    random.seed(seed);

    random_image_paths = random.sample(image_paths, k=n);
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2);
            ax[0].imshow(f);
            ax[0].set_title(f"Original\nSize: {f.size}");
            ax[0].axis("off")

            # Transform and plot image
            # Note: permute() will change shape of image to suit matplotlib 
            # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])

            transformed_image = transform(f).permute(1,2,0);
            ax[1].imshow(transformed_image) 
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)


def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.
    Args:
        dir_path (str or pathlib.Path): target directory

    Returns:
        A print out of:
        number of subdiretories in dir_path
        number of images (files) in each subdirectory
        name of each subdirectory
    """
    try:
        for dirpath, dirnames, filenames in os.walk(dir_path):
            print(
                f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.");
    except:
        print("Incorrect path...");

def main():

    if IMAGE_PATH.is_dir():
        print(f"{IMAGE_PATH} directory exists.");
    else:
        print(f"Did not find {IMAGE_PATH} directory, creating one...")
        IMAGE_PATH.mkdir(parents=True, exist_ok=True);

        # Downloading pizza sushi steak
        with open(DATA_PATH / "pizza_steak_sushi.zip", "wb") as f:
            request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip");
            print("Downloading pizza, steak, sushi data...")
            f.write(request.content)
            
        # Unzip pizza, steak, sushi data
        with zipfile.ZipFile(DATA_PATH / "pizza_steak_sushi.zip", "r") as zip_ref:
            print("Unzipping pizza, steak, sushi data...");
            zip_ref.extractall(IMAGE_PATH);

    # Displays all images in image directory
    # walk_through_dir(dir_path=IMAGE_PATH);

    train_dir = IMAGE_PATH / "train";
    test_dir = IMAGE_PATH / "test";

    random.seed(42);

    # 1. Get all image paths
    image_path_list = list(IMAGE_PATH.glob("*/*/*.jpg"));

    # 2. Get a random image
    random_image_path = random.choice(image_path_list);

    # 3. Get image class from path name (the image class is the name of the directory where the image is stored)
    image_class = random_image_path.parent.stem;

    # 4. Open image
    img = Image.open(random_image_path);


    print(image_class);
    print(img);
    plt.imshow(img);

    # Write transform for an image
    data_transform = transforms.Compose([
        transforms.Resize(size=(64,64)), # resize image to 64x64
        transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance of flipping image horizontally.
        transforms.ToTensor()
    ]);

    
    plot_transformed_images(
        image_path_list, 
        transform=data_transform, 
        n=3
    );

    train_data = datasets.ImageFolder(
        root=train_dir, # target folder of images
        transform=data_transform, # transforms to perform on data (images)
        target_transform=None # transforms to perform on labels (if necessary)
    );

    test_data = datasets.ImageFolder(root=test_dir, transform=data_transform);

    class_names = train_data.classes;
    class_dict = train_data.class_to_idx;

    print(class_names);
    print(class_dict);
    print(len(train_data));
    print(len(test_data));


    torch.manual_seed(22);
    torch.cuda.manual_seed(22);
    random_image_idx = torch.randint(high=len(test_data), size=(1,1)).squeeze().item();
    img, label = test_data[random_image_idx][0], test_data[random_image_idx][1]
    
    img: torch.Tensor = img.permute(1,2,0);

    plt.figure(figsize=(10,7));
    plt.imshow(img);
    plt.axis("off");
    plt.title(class_names[label], fontsize=14);

    train_dataloader = DataLoader( 
        dataset=train_data,
        num_workers=os.cpu_count(), # how many subprocesses to use for data loading
        batch_size=1, # how many samples per batch
        shuffle=True, # shuffle the data?
    );

    test_dataloader = DataLoader(
        dataset=test_data,
        num_workers=os.cpu_count(),
        batch_size=1
    );

    plt.show();
    return 0;


if __name__ == "__main__":
    main();
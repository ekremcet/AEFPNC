import os
import pandas as pd
from skimage.util import random_noise, img_as_ubyte
from skimage import io
from skimage.color import rgb2gray


def crop_image(image, size):
    return image[0:size, 0:size]


def to_gray(image):
    return rgb2gray(image)


def read_img(image, train=False, gray=False):
    img = io.imread(image)
    img = crop_image(img, 256) if train else img
    img = to_gray(img) if gray else img

    return img


def add_gaussian_noise(image, noise_level):
    return random_noise(image, mode="gaussian", seed=None, clip=True, var=(noise_level / 255)**2)


def add_poisson_noise(image):
    return random_noise(image, mode="poisson", seed=None, clip=True)


def prepare_folder(output_folder, set_name, folder_name):
    try:
        new_path = os.path.join(output_folder, set_name, folder_name)
        os.makedirs(new_path)
        return new_path
    except FileExistsError:
        pass


def generate_util_folders():
    try:
        os.makedirs("./Checkpoints")
        os.makedirs("./Samples")
        os.makedirs("./Samples/TestSamples")
    except FileExistsError:
        pass


def generate_gaussian_folder(set_name, data_folder, output_folder, noise_levels, train, gray):
    if train:
        save_path = prepare_folder(output_folder, set_name, "Images")
    for nl in noise_levels:
        if not train:
            save_path = prepare_folder(output_folder, set_name, "gaussian" + str(nl))
        for img_file in os.listdir(data_folder):
            img = read_img(os.path.join(data_folder, img_file), train, gray)
            noisy_img = img_as_ubyte(add_gaussian_noise(img, nl))
            img_name = set_name + "_gaussian" + str(nl) + "_" + img_file[:-4] + ".png"
            io.imsave(os.path.join(save_path, img_name), noisy_img)


def generate_poisson_folder(set_name, data_folder, output_folder, gray):
    save_path = os.path.join(output_folder, set_name, "Images")
    for img_file in os.listdir(data_folder):
        img = read_img(os.path.join(data_folder, img_file), True, gray)
        noisy_img = img_as_ubyte(add_poisson_noise(img))  # Remove extension
        img_name = set_name + "_poisson_" + img_file[:-4] + ".png"
        io.imsave(os.path.join(save_path, img_name), noisy_img)


def generate_ground_truth(set_name, data_folder, output_folder, train, gray):
    save_path = prepare_folder(output_folder, set_name, "GroundTruths")
    for img_file in os.listdir(data_folder):
        img = read_img(os.path.join(data_folder, img_file), train, gray)
        io.imsave(os.path.join(save_path, set_name + "_gt_" + img_file[:-4] + ".png"), img)


def extract_csv_info(filename):
    dataset, noise, index = filename.split("_")
    return dataset, noise, index[:-4]


def generate_csv(noisy_dir, gt_dir, csv_name):
    noisy_data = []
    for _, _, files in os.walk(noisy_dir):
        for filename in sorted(files):
            dataset, noise, index = extract_csv_info(filename)
            entry = {"Dataset": dataset, "Filename": filename, "NoiseLevel": noise, "GT_Index": index}
            noisy_data.append(entry)
    df_noisy = pd.DataFrame(noisy_data)

    gt_data = []
    for _, _, files in os.walk(gt_dir):
        for filename in sorted(files):
            dataset, noise, index = extract_csv_info(filename)
            entry = {"Dataset": dataset, "GT_Index": index, "GT_File": filename}
            gt_data.append(entry)
    df_gt = pd.DataFrame(gt_data)

    # Merge the CSV entries
    df = df_noisy.join(df_gt.set_index(["Dataset", "GT_Index"]), on=["Dataset", "GT_Index"])
    df = df.drop("GT_Index", axis=1)
    df = df.reindex(sorted(df.columns), axis=1)
    df.index.names = ["Index"]
    df.to_csv(csv_name + ".csv", sep=",", encoding="utf-8")

    return df


def generate_csv_files(set_name, output_folder):
    set_folder = os.path.join(output_folder, set_name)
    gt_folder = os.path.join(set_folder, "GroundTruths")
    noisy_folders = []
    for folder in os.listdir(set_folder):
        noisy_folders.append(folder) if folder != "GroundTruths" else ""
    for folder in noisy_folders:
        generate_csv(os.path.join(set_folder, folder), gt_folder, "./" + set_folder + "/" + set_name + "_" + folder)


def generate_noisy_set(data_folder, output_folder, train=False, gray=False):
    set_name = data_folder.split("/")[-1] if not gray else "G" + data_folder.split("/")[-1]
    generate_gaussian_folder(set_name, data_folder, output_folder, [15, 25, 35, 50, 75], train, gray)
    if train:
        generate_poisson_folder(set_name, data_folder, output_folder, gray)
    generate_ground_truth(set_name, data_folder, output_folder, train, gray)
    generate_csv_files(set_name, output_folder)


# Generate CSV for Training Set
generate_csv("./TrainingSet/Images", "./TrainingSet/GroundTruths", "TrainingSet")
# Test Sets
generate_noisy_set("./Datasets/CBSDS68", "./TestSets")
generate_noisy_set("./Datasets/CBSDS68", "./TestSets", gray=True)
generate_noisy_set("./Datasets/Kodak24", "./TestSets")
generate_noisy_set("./Datasets/Set12", "./TestSets")
# Util Folders
generate_util_folders()
import torch.nn as nn
import torch
import time
import os
from DenoisingDataLoader import NoisyDataset, ToTensor
from torch.utils.data import DataLoader
from math import log10
from skimage.measure import compare_psnr, compare_ssim
from torchvision import transforms

BATCH_SIZE = 1  # Required to test with various sized images


def single_ssim(img, gt):
    #  C x H x W -> H x W x C for RGB
    img = img.transpose((1, 2, 0))
    gt = gt.transpose((1, 2, 0))
    ssim = compare_ssim(img, gt, multichannel=True)

    return ssim


def single_psnr(img, gt):
    psnr = compare_psnr(gt, img)

    return psnr


def load_model(model_path):
    model = torch.load(model_path)

    return model


def save_output(model, dataset, sigma, outputs, avg_psnr, avg_ssim, psnrs, ssims, save_ssim=True):
    ct = 0
    dir = os.path.dirname(__file__)
    folder = os.path.join(dir, "Samples", "TestSamples", model.name + "_" + dataset, sigma).replace("\\", "/") + "/"
    if not os.path.exists(folder):
        os.makedirs(folder)

    for img in outputs:
        img = img.cpu().detach().squeeze()
        img = transforms.ToPILImage()(img)
        img.save(folder + str(ct) + ".png")
        ct += 1

    # Save PSNRs and SSIMs
    open(folder + 'AvgPSNR_{:.4f}_dB'.format(avg_psnr), "a").close()
    with open(folder + "psnsrs.txt", "w") as f:
        psnr_idx = 0
        for s in psnrs:
            f.write(str(psnr_idx) + " " + str(s) + "\n")
            psnr_idx += 1

    if save_ssim:
        open(folder + 'AvgSSIM_{:.4f}_dB'.format(avg_ssim), "a").close()
        with open(folder + "ssims.txt", "w") as f:
            ssim_idx = 0
            for s in ssims:
                f.write(str(ssim_idx) + " " + str(s) + "\n")
                ssim_idx += 1


def evaluate(model, dataset, sigma, ssim):
    avg_psnr = 0
    avg_ssim = 0
    avg_time = 0
    outputs = []
    psnrs = []
    ssims = []
    img_transforms = ToTensor()
    testset = NoisyDataset(csv_file="./TestSets/" + dataset + "/" + dataset + "_gaussian" + sigma + ".csv",
                           noisy_dir="./TestSets/" + dataset + "/gaussian" + sigma,
                           gt_dir="./TestSets/" + dataset + "/GroundTruths",
                           transform=img_transforms)
    testset_loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    distance = nn.MSELoss(reduction="mean")
    with torch.no_grad():
        for batch in testset_loader:
            img_batch, gt_batch = batch["image"], batch["gt_img"]
            img_batch = img_batch.cuda().float()
            gt_batch = gt_batch.cuda().float()
            t1 = time.time()
            output = model(img_batch)
            t2 = time.time()
            avg_time += t2-t1
            mse = distance(output, gt_batch).cuda()
            psnr = 10 * log10(1 / mse.item())
            psnrs.append(psnr)
            if ssim:
                ssim = single_ssim(output.cpu().detach().squeeze().numpy(), gt_batch.cpu().detach().squeeze().numpy())
                ssims.append(ssim)
                avg_ssim += ssim
            avg_psnr += psnr
            for out in output:
                outputs.append(out)

        avg_psnr = avg_psnr / len(testset_loader)
        if ssim:
            avg_ssim = avg_ssim / len(testset_loader)
        avg_time = avg_time / len(testset_loader)
        save_output(model, dataset, sigma, outputs, avg_psnr, avg_ssim, psnrs, ssims, save_ssim=ssim)
        print('Average PSNR: {:.4f} dB'.format(avg_psnr))
        if ssim:
            print('Average SSIM: {:.4f}'.format(avg_ssim))
        print('Average Time: {:.4f} s'.format(avg_time))


def evaluate_dataset(model_name, dataset, ssim=True):
    model = load_model("./Checkpoints/" + model_name + ".pth")
    model = model.eval()
    levels = [15, 25, 35, 50, 75]
    for level in levels:
        evaluate(model, dataset, sigma=str(level), ssim=ssim)


if __name__ == '__main__':
    evaluate_dataset("AEFPNC_Gray", "Set12", ssim=False)
    evaluate_dataset("AEFPNC_Gray", "GCBSDS68", ssim=False)
    evaluate_dataset("AEFPNC", "CBSDS68")
    evaluate_dataset("AEFPNC", "Kodak24")

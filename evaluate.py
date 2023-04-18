import lpips
import argparse
import os
import torch
import skimage

import numpy as np
import torchvision.transforms as transforms

from PIL import Image
from glob import glob
from tqdm import tqdm
# from einops import rearrange

def pil_to_numpy(img):
    '''
    Convert PIL image to numpy array and normalize to [0, 1]
    '''
    img = np.array(img)
    # img = rearrange(img, 'h w c -> c h w')
    img = img / 255.0
    return img

def evaluate_psnr(pred, target):
    '''
    Compute PSNR between two images
    '''
    return skimage.metrics.peak_signal_noise_ratio(target, pred, data_range=1.0)

def evaluate_ssim(pred, target):
    '''
    Compute SSIM between two images
    '''
    return skimage.metrics.structural_similarity(target, pred, data_range=1.0, channel_axis=2, multichannel=True)

def evaluate_lpips(pred, target, lpips_fn):
    '''
    Compute LPIPS between two images
    '''
    return lpips_fn(pred, target).item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--fake_data_path", type=str, required=True, help="Path to fake data")
    parser.add_argument("--real_data_path", type=str, required=True, help="Path to real data")
    parser.add_argument("--categories_to_test", nargs='+', type=str, required=True, help="Categories to test")
    parser.add_argument("--size", type=int, default=64, help="Image size to use for evaluation")
    parser.add_argument("--verbose", action="store_true", help="Show progress bar")
    args = parser.parse_args()

    torch.set_grad_enabled(False)

    # Transform to apply to images
    transform = transforms.Compose([transforms.Resize((args.size, args.size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    
    # Load LPIPS model
    lpips_fn = lpips.LPIPS(net='vgg')
    lpips_fn = lpips_fn.cuda()
    lpips_fn.eval()

    # Get categories to test
    categories = args.categories_to_test # [cat1, cat2, ...]

    psnrs = []
    ssims = []
    lpipses = []
    
    # Iterate over categories
    for cat in categories:
        # Get fake and real objects
        fake_objects = glob(os.path.join(args.fake_data_path, f"{cat}_*"))
        real_objects = glob(os.path.join(args.real_data_path, cat, "*"))

        psnr_cat = 0
        ssim_cat = 0
        lpips_cat = 0
        if args.verbose:
            pbar = tqdm(fake_objects)
        else:
            pbar = fake_objects

        # Iterate over objects
        for fake_object in pbar:
            object_id = fake_object.split("/")[-1].split("_")[-1]
            assert os.path.join(args.real_data_path, cat, object_id) in real_objects, f"Object {object_id} not found in real data path"
            fake_views = sorted(glob(os.path.join(fake_object, "0*.png")))
            real_views = sorted(glob(os.path.join(args.real_data_path, cat, object_id, "image", "*.png")))
            assert len(fake_views) == (len(real_views) - 1), f"Number of views mismatched between fake and real data for object {object_id}"

            psnr_object = 0
            ssim_object = 0
            lpips_object = 0
            for fake_view, real_view in zip(fake_views, real_views):
                # Get images
                fake_img = Image.open(fake_view)
                real_img = Image.open(real_view)
                # Convert to numpy and tensor
                fake_numpy = pil_to_numpy(fake_img)
                real_numpy = pil_to_numpy(real_img)
                fake_tensor = transform(fake_img).unsqueeze(0).cuda()
                real_tensor = transform(real_img).unsqueeze(0).cuda()
                # Compute metrics
                psnr_view = evaluate_psnr(fake_numpy, real_numpy)
                ssim_view = evaluate_ssim(fake_numpy, real_numpy)
                lpips_view = evaluate_lpips(fake_tensor, real_tensor, lpips_fn)

                # Add to object metrics
                psnr_object += psnr_view
                ssim_object += ssim_view
                lpips_object += lpips_view
            # Average over views
            psnr_object /= len(fake_views)
            ssim_object /= len(fake_views)
            lpips_object /= len(fake_views)
        # Add to category metrics
        psnr_cat += psnr_object
        ssim_cat += ssim_object
        lpips_cat += lpips_object
        # Average over objects
        psnr_cat /= len(fake_objects)
        ssim_cat /= len(fake_objects)
        lpips_cat /= len(fake_objects)
        
        psnrs.append(psnr_cat)
        ssims.append(ssim_cat)
        lpipses.append(lpips_cat)
        print(f"Category {cat} PSNR: {psnr_cat}, SSIM: {ssim_cat}, LPIPS: {lpips_cat}")
    print(f"Average PSNR: {np.mean(psnrs)}, Average SSIM: {np.mean(ssims)}, Average LPIPS: {np.mean(lpipses)}")

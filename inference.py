import torch.utils.data
import torch
import os
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from train import get_input_dict, norm_batch, get_dice_ji
import cv2
from dataset.tfs import get_lung_transform
import os
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import re
from models.model_single import ModelEmb, ModelEmb3D
from dataset.glas import get_glas_dataset
from dataset.MoNuBrain import get_monu_dataset
from dataset.polyp import get_polyp_dataset, get_tests_polyp_dataset
from dataset.LungData import get_lung_dataset
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from segment_anything.utils.transforms import ResizeLongestSide
import torch.nn.functional as F

sam_args = {
    'sam_checkpoint': "/media/cilab/DATA/Hila/Data/Projects/AutoSAM/sam_vit_h.pth",
    'model_type': "vit_h",
    'generator_args':{
        'points_per_side': 8,
        'pred_iou_thresh': 0.95,
        'stability_score_thresh': 0.7,
        'crop_n_layers': 0,
        'crop_n_points_downscale_factor': 2,
        'min_mask_region_area': 0,
        'point_grids': None,
        'box_nms_thresh': 0.7,
    },
    'gpu_id': 0,
}


def inference_ds(ds, model, sam, transform, epoch, args):

    pbar = tqdm(ds)
    model.eval()
    iou_list = []
    dice_list = []
    Idim = int(args['Idim'])
    NumSliceDim = int(args['NumSliceDim'])
    transform_train, transform_test = get_lung_transform(args)
    count = 0

    for imgs, gts, original_sz, img_sz in pbar:
        count = count + 1
        orig_imgs = imgs.to(sam.device)
        if orig_imgs.ndim == 4:
            orig_imgs = orig_imgs.unsqueeze(0)
        elif orig_imgs.ndim == 3:
            orig_imgs = orig_imgs.unsqueeze(0).unsqueeze(0)
        gts = gts.to(sam.device)

        orig_imgs_small = F.interpolate(
            orig_imgs,
            (Idim, Idim,NumSliceDim),
            mode='trilinear',
            align_corners=True
        )

        orig_imgs_small = orig_imgs_small
        orig_imgs_small = orig_imgs_small.permute(0, 1, 4, 2, 3)
        orig_imgs_small = orig_imgs_small *1/3
        orig_imgs_small = orig_imgs_small.repeat(1, 3, 1, 1, 1)

        gts = gts.permute(0, 3, 1, 2)
        gts_np = gts.squeeze().detach().cpu().numpy()
        plt.imshow(gts_np[35,:,:], cmap="gray")
        plt.show()

        dense_embeddings = model(orig_imgs_small)

        volume_depth = orig_imgs.shape[-1]
        if volume_depth < NumSliceDim:
            print(f"⚠ Volume depth {volume_depth} < NumSliceDim {NumSliceDim}, skipping...")
            continue

        mask = torch.zeros(NumSliceDim, 256, 256)

        for slice_idx in range(NumSliceDim):
            current_slice = orig_imgs_small[:, :, slice_idx, :, :].squeeze()
            current_slice = current_slice.permute(1, 2, 0)
            current_dense_embeddings = dense_embeddings[:, :, slice_idx, :, :]
            gts = gts.squeeze()
            mask_slice = gts[slice_idx, :, :]

            current_slice, mask_slice = transform_test(current_slice.cpu(), mask_slice.cpu())
            original_sz = current_slice.shape[1:3]
            current_slice = transform.apply_image_torch(current_slice)
            current_slice = transform.preprocess(current_slice).cuda()
            img_sz = current_slice.shape[1:3]
            img_sz = torch.tensor(img_sz).unsqueeze(0)
            original_sz = torch.tensor(original_sz).unsqueeze(0)

            batched_input = get_input_dict([current_slice], [original_sz], [img_sz])
            temp_mask = norm_batch(sam_call(batched_input, sam, current_dense_embeddings).unsqueeze(2))

            input_size = tuple([int(x) for x in img_sz[0].squeeze().tolist()])
            original_size = tuple([int(x) for x in original_sz[0].squeeze().tolist()])
            temp_mask = sam.postprocess_masks(temp_mask.squeeze(0), input_size=input_size, original_size=original_size)
            mask_slice = sam.postprocess_masks(mask_slice.unsqueeze(0).unsqueeze(1), input_size=input_size,
                                               original_size=original_size)
            temp_mask = F.interpolate(temp_mask, (Idim, Idim), mode='bilinear', align_corners=True)
            mask_slice = F.interpolate(mask_slice, (Idim, Idim), mode='nearest')

            temp_mask = temp_mask.squeeze().squeeze()
            tensor_min, tensor_max = temp_mask.min(), temp_mask.max()
            mask[slice_idx, :, :] = temp_mask
            temp_mask_np = temp_mask.detach().cpu().numpy()

        mask = mask.unsqueeze(0).unsqueeze(1).cuda()
        scans = orig_imgs.squeeze().squeeze()
        scans = scans.cpu().numpy()

        selected_gts_np = gts.detach().cpu().numpy()
        mask[mask >= 0.5] = 1
        mask[mask < 0.5] = 0

        mask_np = mask.squeeze().squeeze().detach().cpu().numpy()
        fig, axes = plt.subplots(1, 3, figsize=(12, 6))
        scan = scans[:,:,35]
        scan = (scan - scan.min()) / (scan.max() - scan.min())
        axes[0].imshow(scan, cmap="gray")
        axes[0].set_title(f"Test set scan number {count} - slice 35")
        axes[0].axis("off")  # Hide axes


        axes[1].imshow(selected_gts_np[35, :, :], cmap="gray")
        axes[1].set_title("Ground Truth mask")
        axes[1].axis("off")  # Hide axes

        axes[2].imshow(mask_np[35, :, :], cmap="gray")
        axes[2].set_title("Predicted mask")
        axes[2].axis("off")  # Hide axes
        plt.show()


        dice, ji = get_dice_ji(
            mask.squeeze().squeeze().detach().cpu().numpy(),
            gts.detach().cpu().numpy()
        )

        print(f"Dice of scan {count} slice 35: {dice}")
        print(f"IoU of scan {count} slice 35: {ji}")

        iou_list.append(ji)
        dice_list.append(dice)

        pbar.set_description(
            f'(Inference | {args["task"]}) Epoch {epoch} :: Dice {np.mean(dice_list):.4f} :: IoU {np.mean(iou_list):.4f}'
        )

    return np.mean(iou_list)


def sam_call(batched_input, sam, dense_embeddings):
    input_images = torch.stack([sam.preprocess(x["image"]) for x in batched_input], dim=0)
    image_embeddings = sam.image_encoder(input_images)
    sparse_embeddings_none, dense_embeddings_none = sam.prompt_encoder(points=None, boxes=None, masks=None)
    low_res_masks, _ = sam.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings_none,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    return low_res_masks



def main(args=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModelEmb3D(args=args).to(device)
    model1 = torch.load(args['path_best'], map_location=device, weights_only=False)
    model.load_state_dict(model1.state_dict())
    sam = sam_model_registry[sam_args['model_type']](checkpoint=sam_args['sam_checkpoint'])
    sam.to(device=torch.device('cuda', sam_args['gpu_id']))
    transform = ResizeLongestSide(sam.image_encoder.img_size)

    if args['task'] == 'monu':
        trainset, testset = get_monu_dataset(args, sam_trans=transform)
    elif args['task'] == 'glas':
        trainset, testset = get_glas_dataset(sam_trans=transform)
    elif args['task'] == 'polyp':
        trainset, testset = get_polyp_dataset(args, sam_trans=transform)
    elif args['task'] == 'lung':
        trainset, testset = get_lung_dataset(args, sam_trans=transform)

    ds_val = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False,
                                         num_workers=int(args['nW_eval']), drop_last=False)
    with torch.no_grad():
        model.eval()
        inference_ds(ds_val, model.eval(), sam, transform, 0, args)


if __name__ == '__main__':
    # glas 29 256 h
    # monu 34 512 h
    # polyp 56 352 b

    import argparse
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-nW_eval', '--nW_eval', default=0, help='evaluation iteration', required=False)
    parser.add_argument('-task', '--task', default='lung', help='evaluation iteration', required=False)
    parser.add_argument('-depth_wise', '--depth_wise', default=False, help='image size', required=False)
    parser.add_argument('-order', '--order', default=85, help='image size', required=False)
    parser.add_argument('-folder', '--folder', default=474, help='image size', required=False)
    parser.add_argument('-Idim', '--Idim', default=256, help='image size', required=False)
    parser.add_argument('-rotate', '--rotate', default=22, help='image size', required=False)
    parser.add_argument('-NumSliceDim', '--NumSliceDim', default=64, help='image size', required=False)
    parser.add_argument('-scale1', '--scale1', default=0.75, help='image size', required=False)
    parser.add_argument('-scale2', '--scale2', default=1.25, help='image size', required=False)
    args = vars(parser.parse_args())
    base_save_dir = "/media/cilab/DATA/Hila/Data/Projects/AutoSAM/results"
    results_dir = os.path.join(base_save_dir, f'gpu{args["folder"]}')
    args['path_best'] = os.path.join(base_save_dir,
                                     'gpu' + str(args['folder']),
                                     'net_best.pth')
    args['vis_folder'] = os.path.join(base_save_dir, 'gpu' + str(args['folder']), 'vis')
    os.makedirs(args['vis_folder'], exist_ok=True)
    main(args=args)


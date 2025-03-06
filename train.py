

import torch.optim as optim
import torch.utils.data
import torch
import torch.nn as nn
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom
from scipy.ndimage import label
from google.colab import drive
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
# from sam2.build_sam import build_sam2_video_predictor

# from hydra import initialize_config_dir, compose
# from omegaconf import OmegaConf
# from hydra.core.global_hydra import GlobalHydra

def norm_batch(x):
    bs = x.shape[0]
    Isize = x.shape[-1]
    min_value = x.view(bs, -1).min(dim=1)[0].repeat(1, 1, 1, 1).permute(3, 2, 1, 0).repeat(1, 1, Isize, Isize)
    max_value = x.view(bs, -1).max(dim=1)[0].repeat(1, 1, 1, 1).permute(3, 2, 1, 0).repeat(1, 1, Isize, Isize)
    x = (x - min_value) / (max_value - min_value + 1e-6)
    return x


def Dice_loss(y_true, y_pred, smooth=1):
    alpha = 0.5
    beta = 0.5
    tp = torch.sum(y_true * y_pred, dim=(1, 2, 3, 4))
    fn = torch.sum(y_true * (1 - y_pred), dim=(1, 2, 3, 4))
    fp = torch.sum((1 - y_true) * y_pred, dim=(1, 2, 3, 4))
    tversky_class = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return 1 - torch.mean(tversky_class)


def get_dice_ji(predict, target):
    predict = predict + 1
    target = target + 1
    tp = np.sum(((predict == 2) * (target == 2)) * (target > 0))
    fp = np.sum(((predict == 2) * (target == 1)) * (target > 0))
    fn = np.sum(((predict == 1) * (target == 2)) * (target > 0))
    ji = float(np.nan_to_num(tp / (tp + fp + fn)))
    dice = float(np.nan_to_num(2 * tp / (2 * tp + fp + fn)))
    return dice, ji


def open_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)
    a = os.listdir(path)
    os.mkdir(path + '/gpu' + str(len(a)))
    return str(len(a))


def gen_step(optimizer, gts, masks, criterion, accumulation_steps, step):
    size = masks.shape[2:]
    gts_sized = F.interpolate(gts.unsqueeze(dim=1), size, mode='nearest')
    loss = criterion(masks, gts_sized) + Dice_loss(masks, gts_sized)
    loss.backward()
    if (step + 1) % accumulation_steps == 0:  # Wait for several backward steps
        optimizer.step()
        optimizer.zero_grad()
    return loss.item()


def get_input_dict(imgs, original_sz, img_sz):
    batched_input = []
    for i, img in enumerate(imgs):
        if (i==0):
            print(f"len of images: {len(imgs)}")
            print(f"img_sz: {len(img_sz)}")
            print(f"img_sz[i] shape: {img_sz[i].shape}")
            print(i)
        input_size = tuple([int(x) for x in img_sz[i].squeeze().tolist()])
        original_size = tuple([int(x) for x in original_sz[i].squeeze().tolist()])
        singel_input = {
            'image': img,
            'original_size': original_size,
            'image_size': input_size,
            'point_coords': None,
            'point_labels': None,
        }
        batched_input.append(singel_input)
    return batched_input


def postprocess_masks(masks_dict):
    masks = torch.zeros((len(masks_dict), *masks_dict[0]['low_res_logits'].squeeze().shape)).unsqueeze(dim=1).to(device)
    ious = torch.zeros(len(masks_dict)).to(device)
    for i in range(len(masks_dict)):
        cur_mask = masks_dict[i]['low_res_logits'].squeeze()
        cur_mask = (cur_mask - cur_mask.min()) / (cur_mask.max() - cur_mask.min())
        masks[i, 0] = cur_mask.squeeze()
        ious[i] = masks_dict[i]['iou_predictions'].squeeze()
    return masks, ious


def train_single_epoch(ds, model, sam, optimizer, transform, epoch):
    loss_list = []
    pbar = tqdm(ds)
    criterion = nn.BCELoss()
    Idim = int(args['Idim'])
    NumSliceDim = int(args['NumSliceDim'])
    optimizer.zero_grad()
    for ix, (imgs, gts, original_sz, img_sz) in enumerate(pbar):
        orig_imgs = imgs.to(sam.device)
        gts = gts.to(sam.device)
        orig_imgs_small = F.interpolate(orig_imgs, (Idim, Idim), mode='bilinear', align_corners=True)
        dense_embeddings = model(orig_imgs_small)
        batched_input = get_input_dict(orig_imgs, original_sz, img_sz)
        masks = norm_batch(sam_call(batched_input, sam, dense_embeddings))
        loss = gen_step(optimizer, gts, masks, criterion, accumulation_steps=4, step=ix)
        loss_list.append(loss)
        pbar.set_description(
            '(train | {}) epoch {epoch} ::'
            ' loss {loss:.4f}'.format(
                'Medical',
                epoch=epoch,
                loss=np.mean(loss_list)
            ))
    return np.mean(loss_list)

def train_single_epoch3D(ds, model, sam, optimizer, transform, epoch):
    loss_list = []
    pbar = tqdm(ds)
    criterion = nn.BCELoss()
    Idim = int(args['Idim'])
    NumSliceDim = int(args['NumSliceDim'])
    optimizer.zero_grad()

    for ix, (imgs, gts, original_sz, img_sz) in enumerate(pbar):
        orig_imgs = imgs.to(sam.device)
        gts = gts.to(sam.device)

        if orig_imgs.ndim == 4:
            orig_imgs = orig_imgs.unsqueeze(1)
        elif orig_imgs.shape[1] not in [1, 3]:
            orig_imgs = orig_imgs.permute(0, 4, 1, 2, 3)

        orig_imgs_small = F.interpolate(orig_imgs, size=(NumSliceDim, Idim, Idim), mode='trilinear', align_corners=True)
        if orig_imgs_small.shape[1] == 1:
            orig_imgs_small = orig_imgs_small.repeat(1, 3, 1, 1, 1)

        dense_embeddings = model(orig_imgs_small)

        # Perform memory-friendly interpolation
        dense_embeddings = dense_embeddings.float()
        dense_embeddings_split = []
        for i in range(dense_embeddings.shape[0]):
            emb = dense_embeddings[i:i+1]
            emb_interp = F.interpolate(
                emb,
                size=(orig_imgs.shape[2], dense_embeddings.shape[3], dense_embeddings.shape[4]),
                mode='trilinear',
                align_corners=True
            )
            dense_embeddings_split.append(emb_interp)
        dense_embeddings = torch.cat(dense_embeddings_split, dim=0).half()

        batched_input = get_input_dict(orig_imgs, original_sz, img_sz)
        masks = norm_batch(sam_call3D(batched_input, sam, dense_embeddings))

        loss = gen_step(optimizer, gts, masks, criterion, accumulation_steps=4, step=ix)
        loss_list.append(loss)
        pbar.set_description(
            f'(train) epoch {epoch} :: loss {np.mean(loss_list):.4f}'
        )
    return np.mean(loss_list)




def inference_ds(ds, model, sam, transform, epoch, args):
    pbar = tqdm(ds)
    model.eval()
    iou_list = []
    dice_list = []
    Idim = int(args['Idim'])
    NumSliceDim = int(args['NumSliceDim'])
    for imgs, gts, original_sz, img_sz in pbar:
        orig_imgs = imgs.to(sam.device)
        gts = gts.to(sam.device)
        orig_imgs_small = F.interpolate(orig_imgs, (Idim, Idim, Idim), mode='trilinear', align_corners=True)
        dense_embeddings = model(orig_imgs_small)
        batched_input = get_input_dict(orig_imgs, original_sz, img_sz)
        masks = norm_batch(sam_call(batched_input, sam, dense_embeddings))
        input_size = tuple([int(x) for x in img_sz[0].squeeze().tolist()])
        original_size = tuple([int(x) for x in original_sz[0].squeeze().tolist()])
        masks = sam.postprocess_masks(masks, input_size=input_size, original_size=original_size)
        gts = sam.postprocess_masks(gts.unsqueeze(dim=0), input_size=input_size, original_size=original_size)
        masks = F.interpolate(masks, (Idim, Idim, Idim), mode='trilinear', align_corners=True)
        gts = F.interpolate(gts, (Idim, Idim, Idim), mode='nearest')
        masks[masks > 0.5] = 1
        masks[masks <= 0.5] = 0
        dice, ji = get_dice_ji(masks.squeeze().detach().cpu().numpy(),
                               gts.squeeze().detach().cpu().numpy())
        iou_list.append(ji)
        dice_list.append(dice)
        pbar.set_description(
            '(Inference | {task}) Epoch {epoch} :: Dice {dice:.4f} :: IoU {iou:.4f}'.format(
                task=args['task'],
                epoch=epoch,
                dice=np.mean(dice_list),
                iou=np.mean(iou_list)))
    # model.train()
    return np.mean(iou_list)

def sam_call(batched_input, sam, dense_embeddings):
    with torch.no_grad():
        B, C, D, H, W = batched_input.shape  # Get batch & 3D dimensions

        low_res_masks_3D = []  # List to store per-slice masks

        for d in range(D):  # Iterate over depth (slice-by-slice)
            # Extract 2D slice from each 3D volume in batch
            input_slices = torch.stack([sam.preprocess(x["image"][:, :, d, :, :]) for x in batched_input], dim=0)  # Shape: [B, C, H, W]

            # Extract corresponding 2D dense embeddings for this slice
            dense_embeddings_slices = dense_embeddings[:, :, d, :, :]  # Shape: [B, C, H, W]

            # Pass slice through SAM encoder
            image_embeddings = sam.image_encoder(input_slices)
            sparse_embeddings_none, dense_embeddings_none = sam.prompt_encoder(points=None, boxes=None, masks=None)

            # Get low-resolution mask prediction
            low_res_mask_slice, _ = sam.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings_none,
                dense_prompt_embeddings=dense_embeddings_slices,
                multimask_output=False,
            )

            low_res_masks_3D.append(low_res_mask_slice.squeeze(1))  # Store processed mask slice

        # Stack slices back into a full 3D volume
        low_res_masks_3D = torch.stack(low_res_masks_3D, dim=2)  # Shape: [B, 1, D, H, W]

    return low_res_masks_3D


def sam_call(batched_input, sam, dense_embeddings): # Change to sam2
    with torch.no_grad():
        input_images = torch.stack([sam.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = sam.image_encoder(input_images)
        sparse_embeddings_none, dense_embeddings_none = sam.prompt_encoder(points=None, boxes=None, masks=None)
    low_res_masks, iou_predictions = sam.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings_none,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    return low_res_masks


# def sam_call3D(batched_input, sam, dense_embeddings):
#     with torch.no_grad():
#         low_res_masks_3D = []

#         for batch_idx, x in enumerate(batched_input):
#             image = x["image"]  # Shape: [C, D, H, W]
#             C, D, H, W = image.shape
#             mask_slices = []

#             for d in range(D):
#                 slice_img = image[:, d, :, :].unsqueeze(0).float()  # [1, C, H, W]
#                 if slice_img.shape[1] == 1:
#                     slice_img = slice_img.repeat(1, 3, 1, 1)  # convert grayscale to RGB

#                 # Get embeddings as dict and extract
#                 embeddings_dict = sam.image_encoder(slice_img)
#                 image_embeddings = embeddings_dict["image_embeddings"].clone()

#                 sparse_embeddings_none, _ = sam.sam_prompt_encoder(points=None, boxes=None, masks=None)

#                 low_res_mask, _ = sam.sam_mask_decoder(
#                     image_embeddings=image_embeddings,
#                     image_pe=sam.sam_prompt_encoder.get_dense_pe(),
#                     sparse_prompt_embeddings=sparse_embeddings_none,
#                     dense_prompt_embeddings=dense_embeddings[batch_idx, :, d, :, :].unsqueeze(0),
#                     multimask_output=False,
#                     repeat_image=False,  # keep as False for this slice-by-slice approach
#                 )

#                 mask_slices.append(low_res_mask.squeeze(1))

#             full_mask = torch.stack(mask_slices, dim=2)  # [1, H, D, W]
#             full_mask = full_mask.unsqueeze(0)  # Add batch dimension
#             low_res_masks_3D.append(full_mask)

#         result = torch.cat(low_res_masks_3D, dim=0)  # [B, 1, D, H, W]
#         return result

def sam_call3D(batched_input, sam, dense_embeddings):
    with torch.no_grad():
        low_res_masks_3D = []

        for batch_idx, x in enumerate(batched_input):
            image = x["image"]  # Shape: [C, D, H, W]
            C, D, H, W = image.shape
            mask_slices = []

            for d in range(D):
                slice_img = image[:, d, :, :].unsqueeze(0).float()  # [1, C, H, W]
                if slice_img.shape[1] == 1:
                    slice_img = slice_img.repeat(1, 3, 1, 1)  # convert grayscale to RGB

                # Preprocess to the size expected by SAM
                slice_img_resized = sam.preprocess(slice_img.squeeze(0))  # [3, 1024, 1024]
                slice_img_resized = slice_img_resized.unsqueeze(0)  # [1, 3, 1024, 1024]

                # Now get embeddings
                image_embeddings = sam.image_encoder(slice_img_resized)


                # Resize dense embedding slice to match image embedding spatial dims:
                dense_emb_slice = dense_embeddings[batch_idx, :, d, :, :].unsqueeze(0)  # [1, C, H, W]
                dense_emb_slice_resized = F.interpolate(
                    dense_emb_slice,
                    size=(image_embeddings.shape[-2], image_embeddings.shape[-1]),
                    mode='bilinear',
                    align_corners=False
                )

                # Prompt embeddings
                sparse_embeddings_none, _ = sam.prompt_encoder(points=None, boxes=None, masks=None)

                # Decode mask
                low_res_mask, _ = sam.mask_decoder(
                    image_embeddings=image_embeddings,
                    image_pe=sam.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings_none,
                    dense_prompt_embeddings=dense_emb_slice_resized,
                    multimask_output=False,
                )

                mask_slices.append(low_res_mask.squeeze(1))

            full_mask = torch.stack(mask_slices, dim=2)  # [1, H, D, W]
            full_mask = full_mask.unsqueeze(0)  # Add batch dimension
            low_res_masks_3D.append(full_mask)

        result = torch.cat(low_res_masks_3D, dim=0)  # [B, 1, D, H, W]
        return result



def main(args=None, sam_args=None):

    print("Starting main with SAM 2 Video")

    # Set device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = ModelEmb3D(args=args).to(device)
    model = model.half()  # Use half precision for lower memory usage

    # # 🔹 Ensure the config file exists
    # if not os.path.exists(sam_args['config_file']):
    #     raise FileNotFoundError(f"Config file not found: {sam_args['config_file']}")

    # # 🔹 Ensure the checkpoint file exists
    # if not os.path.exists(sam_args['sam_checkpoint']):
    #     raise FileNotFoundError(f"Checkpoint file not found: {sam_args['sam_checkpoint']}")

    # config_dir = os.path.dirname(sam_args['config_file'])  

    # if not os.path.exists(config_dir):
    #     raise FileNotFoundError(f"Config directory not found: {config_dir}")

    # if GlobalHydra.instance().is_initialized():
    #     GlobalHydra.instance().clear()

    # initialize_config_dir(config_dir)
    # cfg = compose(config_name=os.path.basename(sam_args['config_file']).replace(".yaml", ""))
    # print("Loaded Config:", OmegaConf.to_yaml(cfg))

    # sam = build_sam2_video_predictor(
    #     config_file=os.path.basename(sam_args['config_file']).replace(".yaml", ""),
    #     ckpt_path=None,
    #     device=device,
    #     mode="eval",
    #     vos_optimized=True
    # )

    sam = sam_model_registry[sam_args['model_type']](checkpoint=sam_args['sam_checkpoint'])
    sam.to(device=device)

    checkpoint = torch.load(sam_args['sam_checkpoint'], map_location="cpu")
    valid_keys = set(sam.state_dict().keys())
    filtered_checkpoint = {k: v for k, v in checkpoint.items() if k in valid_keys}
    sam.load_state_dict(filtered_checkpoint, strict=False)
    print("Checkpoint successfully loaded into SAM 2.")

    transform = ResizeLongestSide(1024)

    optimizer = optim.Adam(
        model.parameters(),
        lr=float(args['learning_rate']),
        weight_decay=float(args['WD'])
    )

    print('Loading images')
    trainset, testset = get_lung_dataset(args, sam_trans=transform)
    print('Successfully loaded images')

    ds = torch.utils.data.DataLoader(
        trainset,
        batch_size=int(args['Batch_size']),
        shuffle=True,
        num_workers=int(args['nW']),
        drop_last=True)
    
    ds_val = torch.utils.data.DataLoader(
        testset,
        batch_size=1,
        shuffle=False,
        num_workers=int(args['nW_eval']),
        drop_last=False)

    best = 0
    path_best = f'results/gpu{args["folder"]}/best.csv'
    f_best = open(path_best, 'w')

    for epoch in range(int(args['epoches'])):
        print(f"Starting epoch {epoch}")
        train_single_epoch3D(ds, model.train(), sam.eval(), optimizer, transform, epoch)

        with torch.no_grad():
            IoU_val = inference_ds(ds_val, model.eval(), sam, transform, epoch, args)
            if IoU_val > best:
                torch.save(model, args['path_best'])
                best = IoU_val
                print(f"Best results: {best}")
                f_best.write(f"{epoch},{best}\n")
                f_best.flush()

    f_best.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-lr', '--learning_rate', default=0.0003, help='learning_rate', required=False)
    parser.add_argument('-bs', '--Batch_size', default=3, help='batch_size', required=False)
    parser.add_argument('-epoches', '--epoches', default=5000, help='number of epoches', required=False)
    parser.add_argument('-nW', '--nW', default=0, help='evaluation iteration', required=False)
    parser.add_argument('-nW_eval', '--nW_eval', default=0, help='evaluation iteration', required=False)
    parser.add_argument('-WD', '--WD', default=1e-4, help='evaluation iteration', required=False)
    parser.add_argument('-task', '--task', default='glas', help='evaluation iteration', required=False)
    
    parser.add_argument('-dataset_path', '--dataset_path', default='/content/drive/My Drive/Msc/DeepLearning/Project/Task06_Lung/imagesTr', help='Path to the dataset', required=False)
    parser.add_argument('-mask_path', '--mask_path', default='/content/drive/My Drive/Msc/DeepLearning/Project/Task06_Lung/labelsTr', help='Path to the mask dataset', required=False)

    parser.add_argument('-depth_wise', '--depth_wise', default=False, help='image size', required=False)
    parser.add_argument('-order', '--order', default=85, help='image size', required=False)
    parser.add_argument('-Idim', '--Idim', default=512, help='image size', required=False)
    parser.add_argument('-NumSliceDim', '--NumSliceDim', default=32, help='image size', required=False)
    parser.add_argument('-rotate', '--rotate', default=22, help='image size', required=False)
    parser.add_argument('-scale1', '--scale1', default=0.75, help='image size', required=False)
    parser.add_argument('-scale2', '--scale2', default=1.25, help='image size', required=False)
    args = vars(parser.parse_args())
    os.makedirs('results', exist_ok=True)
    folder = open_folder('results')
    args['folder'] = folder
    args['path'] = os.path.join('results',
                                'gpu' + folder,
                                'net_last.pth')
    args['path_best'] = os.path.join('results',
                                     'gpu' + folder,
                                     'net_best.pth')
    args['vis_folder'] = os.path.join('results', 'gpu' + args['folder'], 'vis')
    os.mkdir(args['vis_folder'])
    # sam_args = {
    #     'sam_checkpoint': "/content/sam2/checkpoints/sam2.1_hiera_large.pt",  # ✅ Choose the correct checkpoint
    #     'config_file': "/content/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml",  # ✅ Correct config file
    #     'vos_optimized': True  # ✅ Enable video segmentation mode
    # }

    sam_args = {
        'sam_checkpoint': "/content/drive/My Drive/Msc/DeepLearning/Project/sam_vit_h.pth",
        'model_type': "vit_h",
        'generator_args': {
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

    main(args=args, sam_args=sam_args)


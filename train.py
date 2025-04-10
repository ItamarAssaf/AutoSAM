import torch.optim as optim
import torch.utils.data
import torch
import torch.nn as nn
from tqdm import tqdm
import traceback
import random
from torch.cuda.amp import autocast, GradScaler
from dataset.tfs import get_liver_transform

import os
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom
from scipy.ndimage import label
#from google.colab import drive
import matplotlib.pyplot as plt
import re
from models.model_single import ModelEmb, ModelEmb3D
from dataset.glas import get_glas_dataset
from dataset.MoNuBrain import get_monu_dataset
from dataset.polyp import get_polyp_dataset, get_tests_polyp_dataset
from dataset.LiverData import get_liver_dataset
from segment_anything import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from segment_anything.utils.transforms import ResizeLongestSide
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SingleItemDataset(torch.utils.data.Dataset):
    def __init__(self, sample):
        self.sample = sample

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.sample


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
    # print(f"y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")
    # print(f"y_true min/max: {y_true.min().item()}/{y_true.max().item()}, y_pred min/max: {y_pred.min().item()}/{y_pred.max().item()}")

    tp = torch.sum(y_true * y_pred, dim=(1, 2, 3, 4))
    fn = torch.sum(y_true * (1 - y_pred), dim=(1, 2, 3, 4))
    fp = torch.sum((1 - y_true) * y_pred, dim=(1, 2, 3, 4))
    tversky_class = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return 1 - torch.mean(tversky_class)


def get_dice_ji(predict, target):
    predict = predict.flatten() + 1
    target = target.flatten() + 1
    tp = np.sum((predict == 2) & (target == 2))
    fp = np.sum((predict == 2) & (target == 1))
    fn = np.sum((predict == 1) & (target == 2))
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
    #gts_sized = F.interpolate(gts, size, mode='nearest')
    loss = criterion(masks, gts_sized) + Dice_loss(masks, gts_sized)
    loss.backward()
    if (step + 1) % accumulation_steps == 0:  # Wait for several backward steps
        optimizer.step()
        optimizer.zero_grad()
    return loss.item()


def get_input_dict(imgs, original_sz, img_sz):
    batched_input = []
    for i, img in enumerate(imgs):
        input_size = tuple([int(x) for x in img_sz[i].squeeze().tolist()])
        original_size = tuple([int(x) for x in original_sz[i].squeeze().tolist()])
        #input_size = tuple([int(x) for x in img_sz[i].tolist()])
        #original_size = tuple([int(x) for x in original_sz[i].tolist()])
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


def train_single_epoch3D(ds, model, sam, optimizer, transform, epoch, scaler,sam_trans):
    loss_list = []
    pbar = tqdm(ds)
    criterion = nn.BCEWithLogitsLoss()
    Idim = int(args['Idim'])
    NumSliceDim = int(args['NumSliceDim'])
    optimizer.zero_grad()

    consecutive_failures = 0
    max_failures = 10

    transform_train, transform_test = get_liver_transform(args)

    for ix, batch in enumerate(pbar):
        for sample in batch:
            if consecutive_failures >= max_failures:
                print(f"❌ Stopping training after {max_failures} consecutive failures.")
                return np.mean(loss_list) if loss_list else 0

            try:
                img_tensor, gts, original_sz,img_sz= sample
                gts_np = gts.cpu().numpy()

                orig_img = img_tensor.to(sam.device)
                gts = gts.to(sam.device)

                start_frame = 0
                selected_slices = orig_img[:, :, start_frame:NumSliceDim]
                selected_gts = gts[:, :, start_frame:NumSliceDim]
                selected_slices = selected_slices.permute(2, 0, 1)
                selected_gts = selected_gts.permute(2, 0, 1).unsqueeze(0)

                # Resize and adjust
                orig_imgs_small = F.interpolate(
                    orig_img.unsqueeze(0).unsqueeze(1),  # add batch dim
                    size=(Idim, Idim,NumSliceDim),
                    mode='trilinear', align_corners=True
                )

                orig_imgs_small = orig_imgs_small
                orig_imgs_small = orig_imgs_small.permute(0, 1, 4, 2, 3)
                orig_imgs_small = orig_imgs_small * 1/3
                orig_imgs_small = orig_imgs_small.repeat(1, 3, 1, 1, 1)

                dense_embeddings = model(orig_imgs_small)
                mask = torch.zeros(NumSliceDim, 256, 256)

                for slice_idx in range(NumSliceDim):
                    with torch.cuda.amp.autocast():
                        current_slice = orig_imgs_small[:,:,slice_idx,:,:].squeeze()
                        current_slice = current_slice.permute(1,2,0)
                        mask_slice = selected_gts[:,slice_idx,:,:].squeeze()

                        current_slice, mask_slice = transform_train(current_slice.cpu(), mask_slice.cpu())
                        original_sz = current_slice.shape[1:3]

                        current_slice = sam_trans.apply_image_torch(current_slice)
                        current_slice = sam_trans.preprocess(current_slice).cuda()
                        img_sz = current_slice.shape[1:3]
                        img_sz = torch.tensor(img_sz).unsqueeze(0)
                        original_sz = torch.tensor(original_sz).unsqueeze(0)

                        current_dense_embeddings = dense_embeddings[:,:,slice_idx,:,:]
                        '''
                        dense_embeddings_np = current_dense_embeddings.squeeze().detach().cpu().numpy()
                        plt.imshow(dense_embeddings_np[50,:,:],cmap="gray")
                        plt.title('Dense embedding')
                        plt.show()
                        '''
                        batched_input = get_input_dict([current_slice], [original_sz], [img_sz])
                        temp_mask = norm_batch(sam_call(batched_input, sam, current_dense_embeddings))
                        temp_mask = temp_mask.squeeze().squeeze()
                        mask[slice_idx,  :, :] = temp_mask

                mask = mask.unsqueeze(0).unsqueeze(1).cuda()

                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                selected_gts_np = selected_gts.squeeze().detach().cpu().numpy()

                mask_np = mask.squeeze().squeeze().detach().cpu().numpy()
                mask_np[mask_np >= 0.5] = 1
                mask_np[mask_np < 0.5] = 0

                axes[0].imshow(selected_gts_np[35 ,:, :], cmap="gray")
                axes[0].set_title("Ground Truth mask")
                axes[0].axis("off")  # Hide axes

                axes[1].imshow(mask_np[35, :, :], cmap="gray")
                axes[1].set_title("Predicted mask")
                axes[1].axis("off")  # Hide axes
                plt.show()

                loss = gen_step(optimizer, selected_gts, mask, criterion, accumulation_steps=4, step=ix)
                loss_list.append(loss)

                '''
                # 🔍 Gradient sanity check (every 10 layers only)
                for idx, (name, param) in enumerate(model.named_parameters()):
                    if not param.requires_grad:
                        continue
                    if idx % 10 == 0:
                        if param.grad is not None:
                            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                                print(f"⚠ NaN or Inf detected in gradients of: {name}")
                            elif param.grad.abs().sum() == 0:
                                print(f"⚠ Zero gradients in: {name}")
                        else:
                            print(f"⚠ No gradients for: {name}")
                '''

                # Reset failures after success
                consecutive_failures = 0

            except Exception as e:
                print(f"⚠ Exception encountered: {e}, skipping this sample.")
                consecutive_failures += 1
                continue

        pbar.set_description(f'(train) epoch {epoch} :: loss {np.mean(loss_list):.4f}')

    return np.mean(loss_list)


def inference_ds(ds, model, sam, transform, epoch, args,sam_trans):

    pbar = tqdm(ds)
    model.eval()
    iou_list = []
    dice_list = []
    Idim = int(args['Idim'])
    NumSliceDim = int(args['NumSliceDim'])
    transform_train, transform_test = get_liver_transform(args)


    for imgs, gts, original_sz, img_sz in pbar:
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
        dense_embeddings = model(orig_imgs_small)

        '''
        volume_depth = orig_imgs.shape[-1]
        if volume_depth < NumSliceDim:
            print(f"⚠ Volume depth {volume_depth} < NumSliceDim {NumSliceDim}, skipping...")
            continue
        '''

        mask = torch.zeros(NumSliceDim, 256, 256)

        for slice_idx in range(NumSliceDim):
            current_slice = orig_imgs_small[:, :, slice_idx, :, :].squeeze()
            current_slice = current_slice.permute(1, 2, 0)
            current_dense_embeddings = dense_embeddings[:, :, slice_idx, :, :]
            gts = gts.squeeze()
            mask_slice = gts[slice_idx, :, :]

            current_slice, mask_slice = transform_test(current_slice.cpu(), mask_slice.cpu())
            original_sz = current_slice.shape[1:3]
            current_slice = sam_trans.apply_image_torch(current_slice)
            current_slice = sam_trans.preprocess(current_slice).cuda()
            img_sz = current_slice.shape[1:3]
            img_sz = torch.tensor(img_sz).unsqueeze(0)
            original_sz = torch.tensor(original_sz).unsqueeze(0)

            batched_input = get_input_dict([current_slice], [original_sz], [img_sz])
            temp_mask = norm_batch(sam_call(batched_input, sam, current_dense_embeddings).unsqueeze(2))
            temp_mask = temp_mask.squeeze().squeeze()

            input_size = tuple([int(x) for x in img_sz[0].squeeze().tolist()])
            original_size = tuple([int(x) for x in original_sz[0].squeeze().tolist()])
            temp_mask = sam.postprocess_masks(temp_mask.unsqueeze(0).unsqueeze(1), input_size=input_size, original_size=original_size)
            mask_slice = sam.postprocess_masks(mask_slice.unsqueeze(0).unsqueeze(1), input_size=input_size,
                                               original_size=original_size)
            temp_mask = F.interpolate(temp_mask, (Idim, Idim), mode='bilinear', align_corners=True)
            mask_slice = F.interpolate(mask_slice, (Idim, Idim), mode='nearest')

            temp_mask = temp_mask.squeeze().squeeze()
            mask[slice_idx, :, :] = temp_mask

            '''
            temp_mask_np = temp_mask.detach().cpu().numpy()
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            axes[0].imshow(mask_slice_np, cmap="gray")
            axes[0].set_title("Ground Truth mask")
            axes[0].axis("off")  # Hide axes

            axes[1].imshow(temp_mask_np, cmap="gray")
            axes[1].set_title("Predicted mask")
            axes[1].axis("off")  # Hide axes
            plt.show()
            '''

        mask = mask.unsqueeze(0).unsqueeze(1).cuda()

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        selected_gts_np = gts.detach().cpu().numpy()
        mask[mask >= 0.5] = 1
        mask[mask < 0.5] = 0

        mask_np = mask.squeeze().squeeze().detach().cpu().numpy()
        axes[0].imshow(selected_gts_np[35, :, :], cmap="gray")
        axes[0].set_title("Ground Truth mask")
        axes[0].axis("off")  # Hide axes

        axes[1].imshow(mask_np[35, :, :], cmap="gray")
        axes[1].set_title("Predicted mask")
        axes[1].axis("off")  # Hide axes
        plt.show()

        dice, ji = get_dice_ji(
            mask.squeeze().squeeze().detach().cpu().numpy(),
            gts.detach().cpu().numpy()
        )

        iou_list.append(ji)
        dice_list.append(dice)

        pbar.set_description(
            f'(Inference | {args["task"]}) Epoch {epoch} :: Dice {np.mean(dice_list):.4f} :: IoU {np.mean(iou_list):.4f}'
        )

    return np.mean(iou_list)


def sam_call(batched_input, sam, dense_embeddings):
    with torch.no_grad():
        input_images = torch.stack([sam.preprocess(x["image"]) for x in batched_input], dim=0)
        #input_images = sam.preprocess(batched_input[0]["image"])
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


def main(args=None, sam_args=None):

    scaler = torch.amp.GradScaler()

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    
    # Set device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = ModelEmb3D(args=args).to(device)
    # model = model.half()  # Use half precision for lower memory usage

    
    sam = sam_model_registry[sam_args['model_type']](checkpoint=sam_args['sam_checkpoint'])
    sam.to(device=device)
    sam_trans = ResizeLongestSide(sam.image_encoder.img_size)


    checkpoint = torch.load(sam_args['sam_checkpoint'], weights_only=True, map_location="cpu")
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
    trainset, testset = get_liver_dataset(args, sam_trans=transform)
    print('Successfully loaded images')

    # For debug mode: use only one sample
    if args.get('debug_mode', True):
        single_sample = trainset[0]
        trainset = SingleItemDataset(single_sample)
        print("⚠ Debug mode: training with a single sample only.")

    ds = torch.utils.data.DataLoader(
        trainset,
        batch_size=int(args['Batch_size']),
        shuffle=True,
        num_workers=int(args['nW']),
        drop_last=True,
        collate_fn=lambda x: x
    )

    
    ds_val = torch.utils.data.DataLoader(
        testset,
        batch_size=1,
        shuffle=False,
        num_workers=int(args['nW_eval']),
        drop_last=False)

    # ✅ Set up save directory in Google Drive
    # base_save_dir = "/media/cilab/DATA/Hila/Data/Projects/AutoSAM/results"
    base_save_dir = "/media/cilab/DATA/Hila/Data/Projects/AutoSAM/results"
    os.makedirs(base_save_dir, exist_ok=True)

    results_dir = os.path.join(base_save_dir, f'gpu{args["folder"]}')
    os.makedirs(results_dir, exist_ok=True)

    # ✅ Define best model paths
    args['path_best'] = os.path.join(results_dir, 'net_best.pth')
    path_best = os.path.join(results_dir, 'best.csv')
    args['vis_folder'] = os.path.join(results_dir, 'vis')
    os.makedirs(args['vis_folder'], exist_ok=True)

    best = 0
    # path_best = f'results/gpu{args["folder"]}/best.csv'
    f_best = open(path_best, 'w')

    for epoch in range(int(args['epoches'])):
        print(f"Starting epoch {epoch} out of {args['epoches']}")
        train_single_epoch3D(ds, model.train(), sam.eval(), optimizer, transform, epoch, scaler,sam_trans)

        with torch.no_grad():
            IoU_val = inference_ds(ds_val, model.eval(), sam, transform, epoch, args,sam_trans)
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
    parser.add_argument('-epoches', '--epoches', default=10, help='number of epoches', required=False)
    parser.add_argument('-nW', '--nW', default=0, help='evaluation iteration', required=False)
    parser.add_argument('-nW_eval', '--nW_eval', default=0, help='evaluation iteration', required=False)
    parser.add_argument('-WD', '--WD', default=1e-4, help='evaluation iteration', required=False)
    parser.add_argument('-task', '--task', default='glas', help='evaluation iteration', required=False)
    
    parser.add_argument('-dataset_path', '--dataset_path', default='/media/cilab/DATA/Hila/Data/Projects/AutoSAM/Abdomen/Training/img"', help='Path to the dataset', required=False)
    parser.add_argument('-mask_path', '--mask_path', default=' /media/cilab/DATA/Hila/Data/Projects/AutoSAM/Abdomen/Training/mask', help='Path to the mask dataset', required=False)

    parser.add_argument('-depth_wise', '--depth_wise', default=False, help='image size', required=False)
    parser.add_argument('-order', '--order', default=85, help='image size', required=False)
    parser.add_argument('-Idim', '--Idim', default=256, help='image size', required=False)
    parser.add_argument('-NumSliceDim', '--NumSliceDim', default=64, help='image size', required=False)
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
    args['debug_mode'] = False
    os.mkdir(args['vis_folder'])
    # sam_args = {
    #     'sam_checkpoint': "/content/sam2/checkpoints/sam2.1_hiera_large.pt",  # ✅ Choose the correct checkpoint
    #     'config_file': "/content/sam2/sam2/configs/sam2.1/sam2.1_hiera_l.yaml",  # ✅ Correct config file
    #     'vos_optimized': True  # ✅ Enable video segmentation mode
    # }

    sam_args = {
        #'sam_checkpoint': "/content/drive/My Drive/sam_vit_h.pth",
        'sam_checkpoint': "/media/cilab/DATA/Hila/Data/Projects/AutoSAM/sam_vit_h.pth",
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


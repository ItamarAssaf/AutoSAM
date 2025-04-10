import torch
from monai.networks.nets import SwinUNETR
from monai.transforms import Compose, LoadImage, AddChannel, ScaleIntensity, ToTensor, Resize
from monai.inferers import SlidingWindowInferer
from monai.data import DataLoader, Dataset
from monai.config import print_config
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
import monai
import os
import json
import glob
import pydicom
import numpy as np
import nibabel as nib
import scipy.ndimage
import matplotlib.pyplot as plt
import requests
from tqdm import tqdm
from monai.transforms import (
    AsDiscrete,
    Activationsd,
    AsDiscreted,
    Compose,
    Invertd,
    SaveImaged,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    EnsureTyped,
)
from monai.transforms.utils import allow_missing_keys_mode
from monai.transforms import BatchInverseTransform
from monai.metrics import DiceMetric
from monai.handlers import CheckpointLoader, StatsHandler
#from monai.postprocess import Activationsd, AsDiscreted, Invertd, SaveImaged
from scipy.ndimage import label
from scipy.ndimage import binary_fill_holes

# Create json file
main_data = "/media/cilab/DATA/Hila/Data/Projects/AutoSAM/Abdomen"
Test_data = os.path.join(main_data, "Testing","img")
Train_data = os.path.join(main_data, "Training","img")
json_path  = "/media/cilab/DATA/Hila/Data/Projects/AutoSAM/liver_seg_data.json"
#prediction_mask_path  =

test_data_files = []
train_data_files = []
if not os.path.exists(json_path):
    Test_files = os.listdir(Test_data)
    Train_files = os.listdir(Train_data)
    for file in Test_files:
        path = os.path.join(Test_data, file)
        test_data_files.append({"input": path})
    for file in Train_files:
        path = os.path.join(Train_data, file)
        train_data_files.append({"input": path})

    json_data = {
        "test": test_data_files,
        "train": train_data_files
    }

    # Write the JSON data to a file
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=4)
else:
    with open(json_path, 'r') as f:
        data = json.load(f)


results_path = '/media/cilab/DATA/Hila/Data/Projects/AutoSAM/Abdomen/Testing_prediction_masks_swin_unetr'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if not os.listdir(results_path):
    folder_paths = [entry['input'] for entry in data['test']]

    target_size = (96, 96, 96)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preprocessing_transforms = Compose([
        LoadImaged(keys=["image"], reader="ITKReader", image_only=False),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=[1.5, 1.5, 2.0], mode="bilinear"),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175,
            a_max=250,
            b_min=0.0,
            b_max=1.0,
            clip=True
        ),
        EnsureTyped(keys=["image"])
    ])


    model = SwinUNETR(
        spatial_dims=3,
        in_channels=1,
        out_channels=14,
        img_size=target_size,
        feature_size=48,
        #hidden_size=96,
        #num_layers=4,
        #patch_size=4,
        norm_name="instance",
        use_checkpoint=False
    ).to(device)

    # Inferer setup
    inferer = SlidingWindowInferer(
        roi_size=target_size,
        sw_batch_size=4,
        overlap=0.5
    )

    dataset = Dataset(
        data=[{'image': path} for path in folder_paths],  # Prepare the data format as expected by MONAI Dataset
        transform=preprocessing_transforms
    )

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # Define the path to your weights file
    weights_path = '/media/cilab/DATA/Hila/experiments_and_results/experiments/segmentation_experiments/CT_liver_segmentation/swin_unetr_btcv_segmentation/models/model.pt'
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    postprocessing_transforms = Compose([
        Activationsd(keys="pred", softmax=True),  # Apply softmax to predictions
        AsDiscreted(keys="pred", argmax=True)    # Discretize predictions
    ])

    dice_metric = DiceMetric(include_background=True, reduction="mean")

    def inverse_trasforms(image, output,dataloader):
        batch_infer = output
        batch_infer.applied_operations = image.applied_operations
        segs_dict = {"image": batch_infer}
        batch_inverter = BatchInverseTransform(preprocessing_transforms, dataloader)
        with allow_missing_keys_mode(preprocessing_transforms):
            fwd_bck_batch_labels = batch_inverter(segs_dict)
        output = fwd_bck_batch_labels[0]['image']
        slices = torch.squeeze(output, 0)
        return slices


    # Evaluate the model
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # Move the batch to the GPU
            image = batch["image"].to(device)

            # Perform inference
            outputs = inferer(image, model)
            out = inverse_trasforms(image, outputs,dataloader)
            out = torch.argmax(out, dim=0)
            out = (out ==6)
            values = np.unique(out)
            out = torch.permute(out, (2,0, 1))
            out = torch.transpose(out, 2, 1)
            out = out.cpu().numpy().astype(np.float32)
            values2 = np.unique(out)
            labeled_mask, num_features = label(out)
            img = nib.Nifti1Image(out, affine=np.eye(4))  # Use identity matrix for affine

            meta_data = batch['image_meta_dict']
            folder_path = meta_data['filename_or_obj'][0]
            path_parts = folder_path.split(os.sep)
            case = path_parts[-1]

            output_path = os.path.join(results_path,case)
            nib.save(img, output_path)


# Dice and IoU calculation - ground truth mask and swin_unetr predicted mask
def get_dice_ji(predict, target):
    predict = predict.flatten() + 1
    target = target.flatten() + 1
    tp = np.sum((predict == 2) & (target == 2))
    fp = np.sum((predict == 2) & (target == 1))
    fn = np.sum((predict == 1) & (target == 2))
    ji = float(np.nan_to_num(tp / (tp + fp + fn)))
    dice = float(np.nan_to_num(2 * tp / (2 * tp + fp + fn)))
    return dice, ji


masks = os.listdir(os.path.join(main_data, "Testing","mask"))
predicted_masks_files = os.listdir(results_path)
scans = os.listdir(Test_data)
count = 0
for i in range(len(masks)):
    count = count + 1
    mask = nib.load(os.path.join(main_data,"Testing","mask",masks[i]))
    mask = mask.get_fdata()
    mask = (mask ==6)
    mask = mask.transpose(2, 1, 0)

    prediction_mask = nib.load(os.path.join(results_path, predicted_masks_files[i]))
    prediction_mask = prediction_mask.get_fdata()

    scan = nib.load(os.path.join(Test_data,scans[i]))
    scan = scan.get_fdata()

    fig, ax = plt.subplots(1, 3, figsize=(12,6))
    ax[0].imshow(scan[:,:,90], cmap="gray")
    ax[0].set_title("Scan slice")
    ax[0].axis('off')

    ax[1].imshow(mask[90, :, :], cmap='gray')
    ax[1].set_title("Ground truth mask")
    ax[1].axis('off')

    ax[2].imshow(prediction_mask[90,:,:],cmap='gray')
    ax[2].set_title("Prediction mask")
    ax[2].axis('off')
    plt.tight_layout()
    plt.show()

    dice,ji = get_dice_ji(prediction_mask, mask)

    print(f"Dice score os scan {count}: {dice}")
    print(f"IoU score os scan {count}: {ji}")

# AutoSAM 3D medical image segmentation
Adapting AutoSAM to 3D Medical Images by Overloading the Prompt Encoder

## Overview
This work extend AutoSAM for 3D medical image segmentation (CT scans) by replacing its prompt encoder with a custom encoder tailored for 3D data.

## Datasets
We used the following datasets in our experiments:
[Abdomen data](https://www.synapse.org/#!Synapse:syn3193805/wiki/217752)

## SAM checkopints
[sam base](https://drive.google.com/file/d/1ZwKc-7Q8ZaHfbGVKvvkz_LPBemxHyVpf/view?usp=drive_link)
[sam large](https://drive.google.com/file/d/16AhGjaVXrlheeXte8rvS2g2ZstWye3Xx/view?usp=drive_link)
[sam huge](https://drive.google.com/file/d/1tFYGukHxUCbCG3wPtuydO-lYakgpSYDd/view?usp=drive_link)

## Usage

To use AutoSAM, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/your_username/AutoSAM.git
   cd AutoSAM/

2. conda:

   ```bash
   conda create --name autosam python=3.10
   pip install -r requirements.txt

3. training:
   ```bash
   python train.py

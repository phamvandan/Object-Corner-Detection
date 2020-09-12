## Object-Corner-Detection: Crop an object from four corners
~ By Hisiter-HUST ~ 

![demo_1](demo/result_1.png)


Note:
- This repository can apply to:
  + License Plate Detection 
  + ID Detection 
  + ...

Keywords: Vietnamese License Plate Detection, ID Vietnamese Detection
## 1. Set up environment and dependent package 
- Step 1: Set up conda environment:
```bash
conda env create -f environment.yml
conda activate py36_torch1.4
```
- Step 2: Build DCNv2:

```bash
sh build_dcnv2
```
**Note** 

    * Build successful if output have line ***Zero offset passed***
    * If have any error, try downgrade cudatoolkit  
## 2. Dataset format 
- Prepare custom dataset: [DATA.md](https://github.com/hisiter97/Object-Corner-Detection/blob/master/data/DATA.md)
- Format data: COCO
- Structure
```text

${Object-Corner-Detection_ROOT}
|-- center
|-- data
`-- |-- plate
    `-- |-- annotations
            |-- train_plate.json
            |-- val_plate.json

        |---images
            |--img_1.jpg
            |--img_2.jpg
```
## 3. Training:
- Edit config file at the center/config directory 
```bash
cd center 
python train.py --config config/plate.yml 
```

## 4. Testing:
```bash
cd center
python detect.py --config config/plate.yml --image_path ../img_test/plate.jpg 
```

## 5. DEMO 
![demo_1](demo/result_1.png)

![demo_2](demo/result_2.png)

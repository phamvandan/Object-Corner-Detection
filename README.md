## Object-Corner-Detection: Crop object from four corner
~ By Hisiter-HUST ~

## 1. Set up environment and dependent package 
- At root project, run:  
```bash
sh setup.sh
```

## 2. Dataset format 
- Format COCO
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
- Edit config file at center/config directory 
```bash
cd center 
python train.py --config config/plate.yml 
```

## 4. Testing:
- 
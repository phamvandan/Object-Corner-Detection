### Create Dataset 

####1. Create label by labelImg: https://github.com/tzutalin/labelImg 
- Format VOC 


####2. Convert VOC to COCO (.xml to .json)
- In xml_to_json.py file:
+ Edit from line 101:
    - classes : list 

- Run: python xml_to_json.py
  
    
    
import os
import xml.etree.ElementTree as ET
import pandas as pd


dtype = {
    'classname': 'str',
    'class': 'int'
}
dt_mapping = pd.read_csv('src/player_detection/data/classes.csv', dtype = dtype)



def xml_to_csv(folder_from, save_to='src/player_detection/data/annotations.csv', dt_mapping=dt_mapping):
    
    xml_files = []
    for files in os.listdir(folder_from):
        if files.endswith('.xml'):
            xml_files.append(files)

    # Initialize a pandas df
    dt_csv = pd.DataFrame(
        columns=['image_filename', 'classname', 'x0', 'y0', 'x1', 'y1'])

    # Loop through each XML file
    for xml_file in xml_files:
    
        tree = ET.parse(os.path.join(folder_from, xml_file))
        root = tree.getroot()
    
        for image in root.iter('image'):
            for box in image.iter('box'):
                dt_csv = dt_csv.append(
                    {
                        'image_filename': image.attrib['name'],
                        'classname': box.attrib['label'].replace('-', '_'),
                        'x0': int(float(box.attrib['xtl'])),
                        'y0': int(float(box.attrib['ytl'])),
                        'x1': int(float(box.attrib['xbr'])),
                        'y1': int(float(box.attrib['ybr'])),
                    },
                    ignore_index=True)

    dt_csv = pd.merge(dt_csv, dt_mapping, on='classname', how='left')
    dt_csv.to_csv(save_to, index=False)

if __name__ == '__main__':
    xml_to_csv('/Users/stephanecaron/Downloads/test-xml/')
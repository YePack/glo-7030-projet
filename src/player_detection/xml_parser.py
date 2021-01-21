import glob
import os
import xml.etree.ElementTree as ET
import pandas as pd

dtype = {'classname': 'str', 'class': 'str'}
dt_mapping = pd.read_csv('src/player_detection/data/classes.csv', dtype=dtype)


def xml_to_csv(data_folder, output_file, dt_mapping=dt_mapping):

    xml_files = glob.glob(data_folder + '/*.xml')

    # Initialize a pandas df
    dt_csv = pd.DataFrame(
        columns=['image_id', 'classname', 'xmin', 'ymin', 'xmax', 'ymax'])

    # Loop through each XML file
    for xml_file in xml_files:

        xml_file = xml_file.split('/')[-1]
        tree = ET.parse(os.path.join(data_folder, xml_file))
        root = tree.getroot()

        for image in root.iter('image'):
            for box in image.iter('box'):
                dt_csv = dt_csv.append(
                    {
                        'image_id': image.attrib['name'],
                        'xmin': int(float(box.attrib['xtl'])),
                        'ymin': int(float(box.attrib['ytl'])),
                        'xmax': int(float(box.attrib['xbr'])),
                        'ymax': int(float(box.attrib['ybr'])),
                        'classname': box.attrib['label'].replace('-', '_'),
                    },
                    ignore_index=True)

    dt_csv = pd.merge(dt_csv, dt_mapping, on='classname', how='left')
    dt_csv = dt_csv.astype({
        "xmin": int,
        "xmax": int,
        "ymin": int,
        "ymax": int
    })
    dt_csv = dt_csv.drop(columns=['classname'])
    dt_csv.to_csv(output_file, index=False)


if __name__ == '__main__':
    xml_to_csv('/Users/stephanecaron/Downloads/test-xml/',
               output_file="test.csv")

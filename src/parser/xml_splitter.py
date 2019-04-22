import xml.etree.ElementTree as ET
import os


def split_xml(path_xml):
    tree = ET.parse(path_xml)
    root = tree.getroot()

    path_to = '/'.join(path_xml.split('/')[:-1])
    meta = root[1]

    for image in root.findall('image'):
        filename = image.attrib['name'].replace('.png', '.xml')

        new_root = ET.Element("annotations")
        new_root.append(meta)
        new_root.append(image)

        new_tree = ET.ElementTree(new_root)
        new_tree.write(os.path.join(path_to, filename))


#split_xml('data/raw/multiple_images.xml')
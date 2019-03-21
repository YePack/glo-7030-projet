import xml.etree.ElementTree as ET
import os


def parse_xml_data(path):
    tree = ET.parse(path)
    root = tree.getroot()

    points = []
    labels = []
    for polygon in root.iter('polygon'):
        labels.append(polygon.attrib['label'])
        points_iter = []
        points_list = polygon.attrib['points'].split(';')
        for point in points_list:
            points_iter.append(tuple(map(round, map(float, point.split(',')))))
        points.append(points_iter)

    for polyline in root.iter('polyline'):
        labels.append(polyline.attrib['label'])
        points_iter = []
        points_list = polyline.attrib['points'].split(';')
        for point in points_list:
            points_iter.append(tuple(map(round, map(float, point.split(',')))))
        points.append(points_iter)

    return labels, points


def parse_xml_metadata(path):
    tree = ET.parse(path)
    root = tree.getroot()

    task_name = root.findall('meta')[0][0][1].text
    nb_images = root.findall('meta')[0][0][2].text
    dumped_time = root.findall('meta')[0][1].text
    user = root.findall('meta')[0][0][12][0].text

    return {'task_name': task_name,
            'nb_images': nb_images,
            'dumped_time': dumped_time,
            'user': user}


# Test
#path_xml = os.path.expanduser('./data/xml/test1.xml')
#labels, points = parse_xml_data(path_xml)
#parse_xml_metadata(path_xml)



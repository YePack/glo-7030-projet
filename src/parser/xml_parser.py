import xml.etree.ElementTree as ET
import os


def parse_xml(path):
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


path_xml = os.path.expanduser('./data/xml/test1.xml')
labels, points = parse_xml(path_xml)


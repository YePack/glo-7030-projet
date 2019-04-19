import xml.etree.ElementTree as ET


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
    dumped_time = root.findall('meta')[0][1].text
    user = root.findall('meta')[0][0][12][0].text

    return {'task_name': task_name,
            'dumped_time': dumped_time,
            'user': user}




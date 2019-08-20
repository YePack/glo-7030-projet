import os
import xml.etree.ElementTree as ET

from optparse import OptionParser


def split_xml(path_file, path_to):
	"""
	Function that splits XML of a single job
	into multiple XML (one for each image)
	"""
	
	tree = ET.parse(path_file)
	root = tree.getroot()

	meta = root[1]

	for image in root.findall('image'):
		filename = image.attrib['name'].replace('.png', '.xml')

		new_root = ET.Element("annotations")
		new_root.append(meta)
		new_root.append(image)

		new_tree = ET.ElementTree(new_root)
		new_tree.write(os.path.join(path_to, filename))
		print(filename+' has been created.')


def get_args():
    parser = OptionParser()
    parser.add_option('-f', '--file', type=str, dest='file',
					  help='File Path (including filename) of the XML.')
    parser.add_option('-d', '--dir', type=str, dest='dir',
					  help='Directory to save the XMLs')

    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()

    split_xml(args.file, args.dir)

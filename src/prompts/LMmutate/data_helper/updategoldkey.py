import xml.etree.ElementTree as ET


# 根据原始的gold.key.txt文件针对不同的xml文件，生成对应于其中instance节点的gold.key文件
def generate_new_gold_key(dataset_xml_path, standard_gold_key_path, dataset_gold_key_path):
    # 解析xml文件
    tree = ET.parse(dataset_xml_path)
    root = tree.getroot()

    # 读取txt文件内容:按行读取txt文件，构建列表
    with open(standard_gold_key_path, 'r') as f:
        txt_lines = f.readlines()

    # 获取所有instance节点的id和位置索引
    instance_ids = [instance.attrib['id'] for instance in root.iter('instance')]
    instance_indices = {instance_id: idx for idx, instance_id in enumerate(instance_ids)}

    # 1、根据instance节点的id提取txt文件行
    extract_lines = [line for line in txt_lines if line.split()[0] in instance_ids]

    # 2、根据instance节点的顺序对txt文件行进行排序
    sorted_lines = sorted(extract_lines, key=lambda x: instance_indices[x.split()[0]])

    # 写入新的txt文件
    with open(dataset_gold_key_path, 'w') as f:
        f.writelines(sorted_lines)



if __name__ == '__main__':
    # pass
    dataset = "ALL"
    dataset_dir = "../../../../asset/Evaluation_Datasets/{0}/".format(dataset)
    dataset_xml_path = dataset_dir + "{0}.data.xml".format(dataset)
    dataset_gold_key_path = dataset_dir + "{0}.gold.key.txt".format(dataset)
    standard_gold_key_path = dataset_dir + "{0}_unprocessed.gold.key.txt".format(dataset)
    generate_new_gold_key(dataset_xml_path, standard_gold_key_path, dataset_gold_key_path)
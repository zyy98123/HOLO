import os
import subprocess
from datasets import load_dataset

#1.首先创建fineTuningDataset文件夹
dataset_dir = "fineTuningDataset"
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)
    print(f"Directory '{dataset_dir}' created.")
else:
    print(f"Directory '{dataset_dir}' already exists.")

#2.创建language_only文件夹
language_only_dir = os.path.join(dataset_dir, "Language Only")
if not os.path.exists(language_only_dir):
    os.makedirs(language_only_dir)
    print(f"Directory '{language_only_dir}' created.")
else:
    print(f"Directory '{language_only_dir}' already exists.")

language_only_dir_dataset = [("xingyaoww/code-act", None),
                             ("openbmb/UltraInteract_sft", None),
                             ("stingning/ultrachat", None)]

for dataset_name, version in language_only_dir_dataset:
    if version:
        ds = load_dataset(dataset_name, version, cache_dir=language_only_dir)
        print(f"Dataset {dataset_name} (version {version}) downloaded to {language_only_dir}.")
    else:
        ds = load_dataset(dataset_name, cache_dir=language_only_dir)
        print(f"Dataset {dataset_name} downloaded to {language_only_dir}.")

#3.创建Detailed Image Caption文件夹
detailed_image_caption_dir = os.path.join(dataset_dir, "Detailed Image Caption")
if not os.path.exists(detailed_image_caption_dir):
    os.makedirs(detailed_image_caption_dir)
    print(f"Directory '{detailed_image_caption_dir}' created.")
else:
    print(f"Directory '{detailed_image_caption_dir}' already exists.")

detailed_image_caption_dir_dataset = [("X2FD/LVIS-Instruct4V", None),
                                      ("Lin-Chen/ShareGPT4V", "ShareGPT4V"),
                                      ("Lin-Chen/ShareGPT4V", "ShareGPT4V-PT"),
                                      ("laion/gpt4v-dataset", None),
                                      ]

for dataset_name, version in detailed_image_caption_dir_dataset:
    if version:
        ds = load_dataset(dataset_name, version, cache_dir=detailed_image_caption_dir)
        print(f"Dataset {dataset_name} (version {version}) downloaded to {detailed_image_caption_dir}.")
    else:
        ds = load_dataset(dataset_name, cache_dir=detailed_image_caption_dir)
        print(f"Dataset {dataset_name} downloaded to {detailed_image_caption_dir}.")
#3.1 通过 git clone 下载该数据集，数据集位于repository的data/
VSR_url = "https://github.com/cambridgeltl/visual-spatial-reasoning.git"
try:
    # 执行 git clone 命令，克隆到当前目录
    subprocess.run(["git", "clone", VSR_url], check=True)
    print(f"Successfully cloned {VSR_url}")
except subprocess.CalledProcessError as e:
    print(f"Error occurred while cloning: {e}")

#3.2 创建localized_narratives_dir文件夹，通过wget命令下载该数据集
localized_narratives_dir = os.path.join(detailed_image_caption_dir, "Localized Narratives")
if not os.path.exists(localized_narratives_dir):
    os.makedirs(localized_narratives_dir)
    print(f"Directory '{localized_narratives_dir}' created.")
else:
    print(f"Directory '{localized_narratives_dir}' already exists.")

localized_narratives_dir_urls = ["https://storage.googleapis.com/localized-narratives/annotations/open_images_train_v6_localized_narratives-00000-of-00010.jsonl",
                                 "https://storage.googleapis.com/localized-narratives/annotations/open_images_train_v6_localized_narratives-00001-of-00010.jsonl",
                                 "https://storage.googleapis.com/localized-narratives/annotations/open_images_train_v6_localized_narratives-00002-of-00010.jsonl",
                                 "https://storage.googleapis.com/localized-narratives/annotations/open_images_train_v6_localized_narratives-00003-of-00010.jsonl",
                                 "https://storage.googleapis.com/localized-narratives/annotations/open_images_train_v6_localized_narratives-00004-of-00010.jsonl",
                                 "https://storage.googleapis.com/localized-narratives/annotations/open_images_train_v6_localized_narratives-00005-of-00010.jsonl",
                                 "https://storage.googleapis.com/localized-narratives/annotations/open_images_train_v6_localized_narratives-00006-of-00010.jsonl",
                                 "https://storage.googleapis.com/localized-narratives/annotations/open_images_train_v6_localized_narratives-00007-of-00010.jsonl",
                                 "https://storage.googleapis.com/localized-narratives/annotations/open_images_train_v6_localized_narratives-00008-of-00010.jsonl",
                                 "https://storage.googleapis.com/localized-narratives/annotations/open_images_train_v6_localized_narratives-00009-of-00010.jsonl",
                                 "https://storage.googleapis.com/localized-narratives/annotations/open_images_validation_localized_narratives.jsonl",
                                 "https://storage.googleapis.com/localized-narratives/annotations/open_images_test_localized_narratives.jsonl",
                                 "https://storage.googleapis.com/localized-narratives/annotations/coco_train_localized_narratives-00000-of-00004.jsonl",
                                 "https://storage.googleapis.com/localized-narratives/annotations/coco_train_localized_narratives-00001-of-00004.jsonl",
                                 "https://storage.googleapis.com/localized-narratives/annotations/coco_train_localized_narratives-00002-of-00004.jsonl",
                                 "https://storage.googleapis.com/localized-narratives/annotations/coco_train_localized_narratives-00003-of-00004.jsonl",
                                 "https://storage.googleapis.com/localized-narratives/annotations/coco_val_localized_narratives.jsonl",
                                 "https://storage.googleapis.com/localized-narratives/annotations/flickr30k_train_localized_narratives.jsonl",
                                 "https://storage.googleapis.com/localized-narratives/annotations/flickr30k_val_localized_narratives.jsonl",
                                 "https://storage.googleapis.com/localized-narratives/annotations/flickr30k_test_localized_narratives.jsonl",
                                 "https://storage.googleapis.com/localized-narratives/annotations/ade20k_train_localized_narratives.jsonl",
                                 "https://storage.googleapis.com/localized-narratives/annotations/ade20k_validation_localized_narratives.jsonl"]

for url in localized_narratives_dir_urls:
    wget_command = f"wget '{url}' -P '{localized_narratives_dir}'"
    os.system(wget_command)
    print(f"Downloaded {url.split('/')[-1]} to {localized_narratives_dir}")

#4创建scientific_document_dir
scientific_document_dir = os.path.join(dataset_dir, "Scientific Document")
if not os.path.exists(scientific_document_dir):
    os.makedirs(scientific_document_dir)
    print(f"Directory '{scientific_document_dir}' created.")
else:
    print(f"Directory '{scientific_document_dir}' already exists.")

ds = load_dataset("derek-thomas/ScienceQA", cache_dir=scientific_document_dir)

#5 创建table_dir
table_dir = os.path.join(dataset_dir, "Table")
if not os.path.exists(table_dir):
    os.makedirs(table_dir)
    print(f"Directory '{table_dir}' created.")
else:
    print(f"Directory '{table_dir}' already exists.")

#5.1 IconQA下载
os.system(f"wget \"https://iconqa2021.s3.us-west-1.amazonaws.com/iconqa_data.zip\" -P '{table_dir}'")
print(f"Downloaded {url.split('/')[-1]} to {table_dir}")

os.system(f"wget \"https://drive.google.com/file/d/1iKH2lTi1-QxtNUVRxTUWFvUvRHq6HAsZ/view?usp=sharing\" -P '{table_dir}'")
print(f"Downloaded {url.split('/')[-1]} to {table_dir}")

#5.2 通过 git clone 下载数据集，数据集位于repository的data/tabmwp
tabmwp_url = "https://github.com/lupantech/PromptPG.git"
try:
    subprocess.run(["git", "clone", tabmwp_url], check=True)
    print(f"Successfully cloned {tabmwp_url}")
except subprocess.CalledProcessError as e:
    print(f"Error occurred while cloning: {e}")

chart2text_url = "https://github.com/JasonObeid/Chart2Text.git"
try:
    subprocess.run(["git", "clone", chart2text_url], check=True)
    print(f"Successfully cloned {chart2text_url}")
except subprocess.CalledProcessError as e:
    print(f"Error occurred while cloning: {e}")

table_dir_dataset = [("HuggingFaceM4/ChartQA", None),
                     ("delen/vistext", None),
                     ]
for dataset_name, version in table_dir_dataset:
    if version:
        ds = load_dataset(dataset_name, version, cache_dir=table_dir)
        print(f"Dataset {dataset_name} (version {version}) downloaded to {table_dir}.")
    else:
        ds = load_dataset(dataset_name, cache_dir=table_dir)
        print(f"Dataset {dataset_name} downloaded to {table_dir}.")

#6 创建OCR_dir
OCR_dir = os.path.join(dataset_dir, "Table")
if not os.path.exists(OCR_dir):
    os.makedirs(OCR_dir)
    print(f"Directory '{OCR_dir}' created.")
else:
    print(f"Directory '{OCR_dir}' already exists.")

OCR_dataset = [("Kamizuru00/diagram_image_to_text", None),
               ("lmms-lab/TextCaps", None),
               ("howard-hou/OCR-VQA", None)]

for dataset_name, version in OCR_dataset:
    if version:
        ds = load_dataset(dataset_name, version, cache_dir=OCR_dir)
        print(f"Dataset {dataset_name} (version {version}) downloaded to {OCR_dir}.")
    else:
        ds = load_dataset(dataset_name, cache_dir=OCR_dir)
        print(f"Dataset {dataset_name} downloaded to {OCR_dir}.")

#6.1创建InfographicVQA_dir
InfographicVQA_dir = os.path.join(OCR_dir, "InfographicVQA")
if not os.path.exists(InfographicVQA_dir):
    os.makedirs(InfographicVQA_dir)
    print(f"Directory '{InfographicVQA_dir}' created.")
else:
    print(f"Directory '{InfographicVQA_dir}' already exists.")

InfographicVQA_urls = ["https://rrc.cvc.uab.es/?com=downloads&action=download&ch=17&f=aHR0cHM6Ly9kYXRhc2V0cy5jdmMudWFiLmVzL3JyYy9Eb2NWUUEvVGFzazMvaW5mb2dyYXBoaWNzdnFhX3Fhcy56aXA=",
                       "https://rrc.cvc.uab.es/?com=downloads&action=download&ch=17&f=aHR0cHM6Ly9kYXRhc2V0cy5jdmMudWFiLmVzL3JyYy9Eb2NWUUEvVGFzazMvaW5mb2dyYXBoaWNzdnFhX29jci50YXIuZ3o=",
                       "https://rrc.cvc.uab.es/?com=downloads&action=download&ch=17&f=aHR0cHM6Ly9kYXRhc2V0cy5jdmMudWFiLmVzL3JyYy9Eb2NWUUEvVGFzazMvaW5mb2dyYXBoaWNzdnFhX2ltYWdlcy50YXIuZ3o="]

for url in InfographicVQA_urls:
    wget_command = f"wget '{url}' -P '{InfographicVQA_dir}'"
    os.system(wget_command)
    print(f"Downloaded {url.split('/')[-1]} to {InfographicVQA_dir}")

#6.2创建TextVQA_dir
TextVQA_dir = os.path.join(OCR_dir, "TextVQA_dir")
if not os.path.exists(TextVQA_dir):
    os.makedirs(TextVQA_dir)
    print(f"Directory '{TextVQA_dir}' created.")
else:
    print(f"Directory '{TextVQA_dir}' already exists.")

TextVQA_urls = ["https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_train.json",
                "https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip",
                "https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_Rosetta_OCR_v0.2_train.json",
                "https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json",
                "https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_Rosetta_OCR_v0.2_val.json",
                "https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_test.json",
                "https://dl.fbaipublicfiles.com/textvqa/images/test_images.zip",
                "https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_Rosetta_OCR_v0.2_test.json"]

for url in TextVQA_urls:
    wget_command = f"wget '{url}' -P '{TextVQA_dir}'"
    os.system(wget_command)
    print(f"Downloaded {url.split('/')[-1]} to {TextVQA_dir}")

#7 创建GeneralVQA_dir
GeneralVQA_dir = os.path.join(dataset_dir, "General VQA")
if not os.path.exists(GeneralVQA_dir):
    os.makedirs(GeneralVQA_dir)
    print(f"Directory '{GeneralVQA_dir}' created.")
else:
    print(f"Directory '{GeneralVQA_dir}' already exists.")

GeneralVQA_dataset = [("neuralcatcher/hateful_memes", None),
                      ("ruanchaves/visual7w-gpt", None),
                      ("lmms-lab/GQA", "challenge_all_images"),
                      ("lmms-lab/GQA", "challenge_all_instructions"),
                      ("lmms-lab/GQA", "challenge_balanced_images")]

for dataset_name, version in GeneralVQA_dataset:
    if version:
        ds = load_dataset(dataset_name, version, cache_dir=GeneralVQA_dir)
        print(f"Dataset {dataset_name} (version {version}) downloaded to {GeneralVQA_dir}.")
    else:
        ds = load_dataset(dataset_name, cache_dir=GeneralVQA_dir)
        print(f"Dataset {dataset_name} downloaded to {GeneralVQA_dir}.")

#7.1 下载OK_VQA
OKVQA_dir = os.path.join(GeneralVQA_dir, "OK_VQA")
if not os.path.exists(OKVQA_dir):
    os.makedirs(OKVQA_dir)
    print(f"Directory '{OKVQA_dir}' created.")
else:
    print(f"Directory '{OKVQA_dir}' already exists.")

OKVQA_urls = ["https://okvqa.allenai.org/static/data/mscoco_train2014_annotations.json.zip",
              "https://okvqa.allenai.org/static/data/mscoco_val2014_annotations.json.zip",
              "https://okvqa.allenai.org/static/data/OpenEnded_mscoco_train2014_questions.json.zip",
              "https://okvqa.allenai.org/static/data/OpenEnded_mscoco_val2014_questions.json.zip",
              "http://images.cocodataset.org/zips/train2014.zip",
              "http://images.cocodataset.org/zips/val2014.zip"]
for url in OKVQA_urls:
    wget_command = f"wget '{url}' -P '{OKVQA_dir}'"
    os.system(wget_command)
    print(f"Downloaded {url.split('/')[-1]} to {OKVQA_dir}")

#7.2下载AOKVQA
AOKVQAurl = "https://prior-datasets.s3.us-east-2.amazonaws.com/aokvqa/aokvqa_v1p0.tar.gz"
AOK_wget_command = f"wget '{AOKVQAurl}' -P '{GeneralVQA_dir}'"
os.system(AOK_wget_command)
print(f"Downloaded {AOKVQAurl.split('/')[-1]} to {GeneralVQA_dir}")

#7.3 下载TallyQA 
os.system(f"wget \"https://github.com/manoja328/tallyqa/blob/master/tallyqa.zip?raw=true\" -P '{GeneralVQA_dir}'")

#7.4 下载COCO-QA
os.system(f"wget \"http://www.cs.toronto.edu/~mren/imageqa/data/cocoqa/cocoqa-2015-05-17.zip\" -P '{GeneralVQA_dir}'")

#7.5 下载VQAV2
VQAV2_dir = os.path.join(GeneralVQA_dir, "VQAV2")
if not os.path.exists(VQAV2_dir):
    os.makedirs(VQAV2_dir)
    print(f"Directory '{VQAV2_dir}' created.")
else:
    print(f"Directory '{VQAV2_dir}' already exists.")

VQAV2_urls = ["https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip",
              "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip",
              "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip",
              "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip",
              "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip",
              "http://images.cocodataset.org/zips/train2014.zip",
              "http://images.cocodataset.org/zips/val2014.zip",
              "http://images.cocodataset.org/zips/test2015.zip",
              "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Complementary_Pairs_Train_mscoco.zip",
              "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Complementary_Pairs_Val_mscoco.zip"]

for url in VQAV2_urls:
    wget_command = f"wget \"{url}\" -P '{VQAV2_dir}'"
    os.system(wget_command)
    print(f"Downloaded {url.split('/')[-1]} to {VQAV2_urls}")




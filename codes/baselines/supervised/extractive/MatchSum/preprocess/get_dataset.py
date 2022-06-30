import os
import json
import numpy as np
from tqdm import tqdm

def preproces(s_):
  return [line.replace('\n',"").replace("  "," ") for line in s_]

splits = ["train", "test", "val"] 
for split in splits:
    base_path = "/content/Long-Text-Summarization/data/final/exp1"
    path_articles = f"{base_path}/{split}/ects"
    path_summaries = f"{base_path}/{split}/gt_summaries"
    path_indices = f"/content/Long-Text-Summarization/PreSumm/results/selected_ids/"
    outfile = f"/content/Long-Text-Summarization/MatchSum/data/ect_data_{split}.jsonl"
    outfile2 = f"/content/Long-Text-Summarization/MatchSum/data/ect_index_{split}.jsonl"

    articles = os.listdir(path_articles)
    summaries = os.listdir(path_summaries)
    print(split, len(articles), len(summaries))
    
    data = []
    indices = []
    missing_cnt=0
    for article in tqdm(articles):
        data_point = {}
        index = {}
        a_ = open(os.path.join(path_articles, article), 'r').readlines()
        a = preproces(a_)
        s_ = open(os.path.join(path_summaries, article), 'r').readlines()
        s = preproces(s_)
        try:
          i_ = np.load(os.path.join(path_indices, str(article)[:-4] + ".npy"), 'r')
        except:
          missing_cnt=missing_cnt+1
          pass
          
        #skip empty files
        if len(a_)==0 or len(s_)==0:
          os.remove(os.path.join(path_articles, article))
          os.remove(os.path.join(path_summaries, article))
          print(f"Deleting {article} as it is empty")

        data_point["text"] = a
        data_point["summary"] = s
        data_point["article_id"] = article
        index["sent_id"] = (i_).tolist()[0]
        data.append(data_point)
        indices.append(index)

    print("\nmissing_cnt: ", missing_cnt, "split: ", split)
    with open(outfile, "w") as myfile:
        for d in data:
          myfile.write(json.dumps(d) + "\n")          
        print(f"{split} data done")
    myfile.close()

    with open(outfile2, "w") as myfile2:
        for v in indices:
          myfile2.write(json.dumps(v) + "\n")          
        print(f"{split} index done")
    myfile2.close()
from transformers import CLIPProcessor, CLIPModel
import torch
import json
from operator import itemgetter
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt

# Using the clip-vit-base-patch32 model from openai
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# images features of clip.json
show = "tagesschau"
day = "20220227"
f = open(os.path.join(show, day, "clip.json"))
data_clip = json.load(f)

f = open(os.path.join(show, day, "blip.json"))
data_blip = json.load(f)


def cos_sim_clip_blip(data_clip, data_blip):
    image = data_clip['y']
    caption = data_blip['text']

    image_features = torch.tensor(image)
    # add new dimension at position 0 to match shape of text tensor
    image_features = image_features.unsqueeze(0)
    image_features /= torch.linalg.norm(image_features, ord=2, dim=-1, keepdim=True)
    # image_features /= image_features.norm(p=2, dim=-1, keepdim=True)

    # blip caption
    text_input = processor(text=caption, return_tensors="pt")
    text_features = model.get_text_features(**text_input)
    text_features /= torch.linalg.norm(text_features, ord=2, dim=-1, keepdim=True)
    # text_features /= text_features.norm(p=2, dim=-1, keepdim=True)

    # compute cosine similarity
    logit_scale = model.logit_scale.exp()
    similarity = torch.nn.functional.cosine_similarity(text_features, image_features) * logit_scale

    return {
        "score": similarity.item(),
        "label": caption,
        "timestamp": data_clip['t']
    }


def iterate_over_data(data_clip, data_blip):
    results = []
    i = 0
    for index, blip_caption in enumerate(data_blip):
        results_of_timeslot = []
        # sometimes there is a time gap between the captions of blip
        # therefore we need to step over the clip embedding
        while data_clip[i]['t'] < blip_caption['begin']:
            i += 1
        while blip_caption['begin'] <= data_clip[i]['t'] <= blip_caption['end']:
            if blip_caption['begin'] <= data_clip[i]['t'] <= blip_caption['end']:
                res = cos_sim_clip_blip(data_clip[i], blip_caption)
                results_of_timeslot.append(res)
                i += 1
        # just add the clib-emb and blip-caption combination with the best score
        if len(results_of_timeslot) == 1:
            results.append(results_of_timeslot[0])
        elif len(results_of_timeslot) > 0:
            res_sorted = rank_scores(results_of_timeslot)
            results.append(res_sorted[0])
    return results


# Sorting results by score in descending order
def rank_scores(results):
    return sorted(results, key=itemgetter('score'), reverse=True)


def write_to_csv(results):
    filename = 'results_' + day + ".csv"
    filepath = os.path.join(show, day, filename)
    with open(filepath, 'w', encoding='utf8', newline='') as output_file:
        fc = csv.DictWriter(output_file,
                            fieldnames=results[0].keys(),
                            )
        fc.writeheader()
        fc.writerows(results)



def main():
    # creating results.csv from a blip.json and a clip.json with their cosine similarity score
    # res = iterate_over_data(data_clip, data_blip)
    # res_ranked = rank_scores(res)
    # print(res_ranked)
    # write_to_csv(res_ranked)

    # concatenate .csv to one big .csv for bildtv and tagesschau
    # filenames = [
    #     "bildtv/20220127_Ukraine_konflikt_Ukraine_braucht_Defensiv/results_20220127.csv",
    #     "bildtv/20220214_Ukraine_Konflikt_droht_Eskalation/results_20220214.csv",
    #     "bildtv/20220215_Krimi_um_Krieg_und_Frieden/results_20220215.csv",
    #     "bildtv/20220221_Putin_geht_imemr_einen_Schritt/results_20220221.csv",
    #     "bildtv/20220222_Sorge_vor_einem_großen_Krieg/results_20220222.csv",
    #     "bildtv/20220224_Russland_Angriff_Seit_heute_Nacht/results_20220224.csv",
    #     "bildtv/20220225_Kampf_um_Kiew_Bürger_wollen/results_20220225.csv",
    #     "bildtv/20220226_Raketen_Einschlag_In_Kiew/results_20220226.csv"
    # ]

    # filenames = [
    #     "tagesschau/20220127/results_20220127.csv",
    #     "tagesschau/20220214/results_20220214.csv",
    #     "tagesschau/20220221/results_20220221.csv",
    #     "tagesschau/20220222/results_20220222.csv",
    #     "tagesschau/20220224/results_20220224.csv",
    #     "tagesschau/20220225/results_20220225.csv",
    #     "tagesschau/20220226/results_20220226.csv",
    #     "tagesschau/20220227/results_20220227.csv",
    # ]

    # Create combinded csv of all files in filenames
    # combined_csv = pd.concat([pd.read_csv(f) for f in filenames])
    # combined_csv.to_csv("tagesschau/results_combined.csv", index=False)

    #pandas section create awesome diagramms for visualisation
    df = pd.read_csv("bildtv/results_combined_bildtv.csv")
    score_column = df["score"]

    # setting font size to 30
    plt.rcParams.update({'font.size': 14})
    # create histogram
    score_column.plot(kind="hist", color='lightblue', ec="black")
    # setting labels
    plt.xlabel("Kosinus-Ähnlichkeits-Wert")
    plt.ylabel("Häufigkeit")

    #plt.show()
    plt.savefig('verteilung-der-scores.png', bbox_inches='tight')


if __name__ == "__main__":
    main()

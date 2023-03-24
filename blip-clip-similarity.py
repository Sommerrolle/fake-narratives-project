from transformers import CLIPProcessor, CLIPModel
import torch
import json
from operator import itemgetter
import os
import csv

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
    res = iterate_over_data(data_clip, data_blip)
    res_ranked = rank_scores(res)
    print(res_ranked)
    write_to_csv(res_ranked)


if __name__ == "__main__":
    main()

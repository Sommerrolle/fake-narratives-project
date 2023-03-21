from transformers import CLIPProcessor, CLIPModel
import torch
import json

# Using the clip-vit-base-patch32 model from openai
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# images features of clip.json
f = open("clip.json")
data_clip = json.load(f)

f = open("blip.json")
data_blip = json.load(f)

def cos_sim_clip_blip(data_clip, data_blip):

    image = data_clip['y']
    caption = data_blip['text']

    image_features = torch.tensor(image)
    # add new dimension at position 0 to match shape of text tensor
    image_features = image_features.unsqueeze(0)
    image_features /= torch.linalg.norm(image_features, ord=2, dim=-1, keepdim=True)
    #image_features /= image_features.norm(p=2, dim=-1, keepdim=True)

    # blip caption
    text_input = processor(text=caption, return_tensors="pt")
    text_features = model.get_text_features(**text_input)
    text_features /= torch.linalg.norm(text_features, ord=2, dim=-1, keepdim=True)
    #text_features /= text_features.norm(p=2, dim=-1, keepdim=True)

    # compute cosine similarity
    # funktioniert aber kommen komische Ergenisse raus
    logit_scale = model.logit_scale.exp()
    similarity = torch.nn.functional.cosine_similarity(text_features, image_features) * logit_scale


    return [{
        "score": similarity.item(),
        "label": caption,
        "timestamp": data_clip['t']
    }]

    # sim = image_features.detach().numpy() @ text_features.detach().numpy().T
    # return (sim * 100).softmax(dim=1)[0]

    # geht auch mit der funktion
    #similarity_score = torch.nn.functional.cosine_similarity(image_features, text_features)
    #return similarity_score.item() * 100


def iterate_over_data(data_clip, data_blip):
    results = []
    i = 0
    for i in data_clip:
        if data_blip[i]['begin'] <= data_clip[i]['t'] <= data_blip[i]['end']:
            res = cos_sim_clip_blip(data_clip[i], data_blip[i])
            results.append(res)


def main():
    image_features = data_clip[0]
    caption = data_blip[0]
    result = cos_sim_clip_blip(image_features, caption)
    print('result:')
    print(result)


if __name__=="__main__":
    main()



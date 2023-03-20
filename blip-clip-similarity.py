from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
import torch
import json

# function to calculate the cosine similarity of
# the clip embedding and the blip caption

# Using the clip-vit-base-patch32 model from openai
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# images features of clip.json
f = open("clip.json")
data_clip = json.load(f)

f = open("blip.json")
data_blip = json.load(f)

def cos_sim_clip_blip(image, caption):
    # clip image embedding
    # convert json data in tensor
    # data = []
    # for i in image:
    #     if type(i) == str:
    #         float(i)
    #     else:
    #         data.append(i)

    captions_list = []
    captions_list.append(caption)
    captions_list.append("dog in front of a house")

    image_features = torch.tensor(image)
    # add new dimension at position 0 to match shape of text tensor
    image_features = image_features.unsqueeze(0)
    print(image_features.shape)
    image_features /= torch.linalg.norm(image_features, ord=2, dim=1, keepdim=True)
    print(image_features)

    # blip caption
    text_input = processor(text=captions_list, return_tensors="pt")
    text_feature = model.get_text_features(**text_input)
    text_feature /= torch.linalg.norm(text_feature, ord=2, dim=1, keepdim=True)
    print(text_feature)
    print(text_feature.shape)

    # calculate cos similarity

    sim = torch.tensor([[torch.dot(x, image_features[0]) for x in text_feature]])
    scores = (sim * 100).softmax(dim=1)[0]

    return [{
        "score": scores[i].item(),
        "label": label
    } for i, label in enumerate(captions_list)]


def main():
    image_features = data_clip[0]['y']
    caption = data_blip[0]['text']
    result = cos_sim_clip_blip(image_features, caption)
    print('result:')
    print(result)


if __name__=="__main__":
    main()



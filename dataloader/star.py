import torch
from .base_dataset import BaseDataset
import json

class STAR(BaseDataset):
    def __init__(self, args=None, tokenizer=None, split='train'):
        super().__init__(args, tokenizer, split)
        self.data = json.load(open(f'./data/star/STAR_{split}.json', 'r'))
        self.features = torch.load(f'./data/star/clipvitl14.pth')
        self.answer_mapping = {0: '(A)', 1: '(B)', 2: '(C)', 3: '(D)'}
        self.qtype_mapping = {'Interaction': 1, 'Sequence': 2, 'Prediction': 3, 'Feasibility': 4}
        self.num_options = 4
        print(f"Num {split} data: {len(self.data)}") 


    def _get_text(self, idx):
        question = self.data[idx]["question"].capitalize().strip()
        if question[-1] != "?":
            question = str(question) + "?"
            
        options = {x['choice_id']: x['choice'] for x in self.data[idx]['choices']}
        options = [options[i] for i in range(self.num_options)]
        answer = options.index(self.data[idx]['answer'])
        
        q_text = f"Question: {question}\n"
        o_text = "Choices: \n"
        for i in range(self.num_options):
            o_text += f"{self.answer_mapping[i]} {options[i]}\n"
        a_text = "Answer: The answer is "
        text = {'q_text': q_text, 'o_text': o_text, 'a_text': a_text, 'options': options}
        return text, answer

    def get_video_with_indices(self, video_id, start, end):
        if video_id not in self.features:
            print(video_id)
            video = torch.zeros(1, self.features_dim)
            original_indices = [0]
        else:
            video = self.features[video_id][start: end + 1, :].float()  # ts
            original_indices = list(range(start, end + 1))
        
        if len(video) > self.max_feats:
            sampled = []
            sampled_indices = []
            for j in range(self.max_feats):
                idx = (j * len(video)) // self.max_feats
                sampled.append(video[idx])
                sampled_indices.append(original_indices[idx])
            video = torch.stack(sampled)
            video_len = self.max_feats
            return_indices = sampled_indices
        elif len(video) < self.max_feats:
            video_len = len(video)
            video = torch.cat([video, torch.zeros(self.max_feats - video_len, self.features_dim)], 0)
            # Pad indices with -1 to indicate padding frames
            return_indices = original_indices
        else:
            video_len = self.max_feats
            return_indices = original_indices
        
        return video, video_len, return_indices
    
    def __getitem__(self, idx):
        vid = self.data[idx]['video_id']
        question_id = self.data[idx]['question_id']
        qtype = self.qtype_mapping[self.data[idx]['question_id'].split('_')[0]]
        text, answer = self._get_text(idx)
        text_id, label, video_start, video_index, label_mask = self._get_text_token(text, answer)
        start, end = round(self.data[idx]['start']), round(self.data[idx]['end'])
        video, video_len, frames = self.get_video_with_indices(f'{vid}', start, end)
        return {"vid": vid, "video": video, "video_len": video_len, "text": text, "text_id": text_id, "label": label, "video_start": video_start,
                "video_index": video_index, "label_mask": label_mask, "qid": idx, "question_id": question_id, "answer": answer, "qtype": qtype, "frames": frames}


    def __len__(self):
        return len(self.data)
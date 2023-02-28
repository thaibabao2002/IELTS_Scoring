import numpy as np
import onnxruntime as ort
from scipy.special import softmax
from transformers import AutoTokenizer
import torch

class IeltsONNXPredictor:
    def __init__(self, model_path_coherence="save_model/model_coherence.onnx", model_path_task="save_model/model_task.onnx", model_name = "bert-base-uncased"):
        self.ort_session_task = ort.InferenceSession(model_path_task)
        self.ort_session_coherence = ort.InferenceSession(model_path_coherence)
        self.label_task = [4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5,8, 8.5, 9]
        self.label_coherence = [4, 5, 6, 7, 8, 9]
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    def predict(self, essay, input):
        encoder_coh = self.tokenizer(
            essay,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        encoder_task = self.tokenizer(
            input,
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        ort_inputs_task = {'input_ids':encoder_task['input_ids'], 'attention_mask':encoder_task['attention_mask']} 
        ort_inputs_task = {input_name: np.array(input_tensor, dtype=np.int64) for input_name, input_tensor in ort_inputs_task.items()}
        ort_outs_task = self.ort_session_task.run(None, ort_inputs_task)
        scores_task = softmax(ort_outs_task[0])[0]
        pred_task = self.label_task[np.argmax(scores_task)]

        ort_inputs_coh = {'input_ids':encoder_coh['input_ids'], 'attention_mask':encoder_coh['attention_mask']} 
        ort_inputs_coh = {input_name: np.array(input_tensor, dtype=np.int64) for input_name, input_tensor in ort_inputs_coh.items()}
        ort_outs_coh = self.ort_session_coherence.run(None, ort_inputs_coh)
        scores_coh = softmax(ort_outs_coh[0])[0]
        pred_coh = self.label_coherence[np.argmax(scores_coh)]

        return pred_task, pred_coh

if __name__ == "__main__":
    question = "some people say music is a good way of bringing people of different culture and ages together. to what extent do you agree or disagree"
    essay = "in this modern era of technology, music plays a significant role in gathering the population. some communities suggest that it is crucial for the closeness of different traditions all over the world. i suggest that music is vital but there are ample other things included in combining the norms of a society. to commence with, it is really proliferating these modern days as a result of advancements in technology. firstly, it depicts the rituals of one's state and attracts a plethora of folk including civilisations, behaviours and current issues of the country as well.therefore, by listening to unusual operas people want to know more about the inhabitants of the territory and their relationship gets stronger by interacting with each other. for example, the korean pop music band named bts expresses korean culture by illustrating dance, and pop tunes and has millions of followers from all over the globe. on the other hand, some people face problems in understanding the lyrics of the song because of the different language barriers of the state. furthermore, business and also travel documentaries play a huge role in this regard that represent the foods, cultures, and attitudes of the public of a particular state. consequently, they get fascinated by a large majority of the people. to illustrate an example, the travel vlogger named daud kim explores captivating places of many states and vlogs them in a spectacular way that gets society's attention to know more about their norms. in conclusion, in light of these convincing arguments, i believe that opera along with travel and trade are the essential factors for mixing up the individuals of special traditions."
    input = 'CLS '+ question + ' SEP ' + essay + ' SEP'

    predictor = IeltsONNXPredictor()
    print(predictor.predict(essay, input))
from aes_function.layers import Conv1DMask, GatePositional, MaxPooling1DMask
from aes_function.layers import MeanOverTime
from aes_function.reader import process_essay, convert_to_dataset_friendly_scores
from keras.models import model_from_json


class Model:
    def __init__(self):
        self.custom_objects = {'Conv1DMask': Conv1DMask,
                               'GatePositional': GatePositional,
                               'MaxPooling1DMask': MaxPooling1DMask,
                               'MeanOverTime': MeanOverTime}

    def calculate_score(self, essay):
        json_file = open('model/model.json', 'r')
        model_json = json_file.read()
        json_file.close()
        model = model_from_json(model_json, custom_objects=self.custom_objects)
        input_w_shape = model.get_layer('input_word').output_shape
        input_c_shape = model.get_layer('input_char').output_shape
        model.load_weights('model/weights.h5')
        predictions = model.predict(
            process_essay(essay, input_w_shape, input_c_shape),
            batch_size=180).squeeze()
        score = convert_to_dataset_friendly_scores(predictions)
        return score

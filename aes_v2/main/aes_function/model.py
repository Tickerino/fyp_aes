from main.aes_function.layers import Conv1DMask, GatePositional, MaxPooling1DMask
from main.aes_function.layers import MeanOverTime
from main.aes_function.reader import process_essay, convert_to_dataset_friendly_scores
from keras.models import model_from_json

class Model:
    def __init__(self):
        self.custom_objects = {'Conv1DMask': Conv1DMask,
                               'GatePositional': GatePositional,
                               'MaxPooling1DMask': MaxPooling1DMask,
                               'MeanOverTime': MeanOverTime}

    def calculate_score(self, essay, question):

        weights_path = 'main/model/weights{}.h5'.format(question.question_id)
        model_path = 'main/model/model{}.json'.format(question.question_id)

        try:
            json_file = open(model_path,'r')
        except IOError:
            return -1

        with json_file:
            model_json = json_file.read()
            json_file.close()
            model = model_from_json(model_json, custom_objects=self.custom_objects)
            input_w_shape = model.get_layer('input_word').output_shape
            input_c_shape = model.get_layer('input_char').output_shape
            model.load_weights(weights_path)
            predictions = model.predict(
                process_essay(essay, input_w_shape, input_c_shape),
                batch_size=180).squeeze()
            score = convert_to_dataset_friendly_scores(predictions)
            return score

import tensorflow as tf
import numpy as np
class Model:
    
    def __init__(self):
        self.model = tf.keras.models.load_model('model.keras')
        # Load the tokenizer (using the same tokenizer used in training the model)
        self.tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(open('tokenizer.json').read())
    
    def predict(self, string):
        # Preprocess the input
        sequence = self.tokenizer.texts_to_sequences([string])
        padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=50) # 50 was the max length of the sequences used in training the model
        # Make the prediction
        output = self.model.predict(padded_sequence)
        # Define the labels
        labels = ['Negative', 'Neutral', 'Positive']
        # Get the index of the maximum value in the output array
        max_index = np.argmax(output)
        # Return the corresponding label
        return labels[max_index]
        


# Test
if __name__ == '__main__':
    model = Model()
    pos =  "كوتش انتا انسان محترم او عندك قلب كبير نشوفك غير فتلفزيون بصح مشاء الله تبارك الله" 
    neg = "حمار مستحمر يرأس اجتماع حمير مستنفرة"
    nue = "نقدر نروح للجامعة ديركت بسك كل يوم و انا رايحة جاية بغيت نروح ندفع اوراقي دربة وحدة؟؟"

    print(model.predict(pos)) # Output: Should be Positive
    print(model.predict(neg)) # Output: Should be Negative
    print(model.predict(nue)) # Output: Should be Neutral
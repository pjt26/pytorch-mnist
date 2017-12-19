import numpy
import torch
from torch.autograd import Variable
from torchvision import transforms
from flask import Flask, jsonify, render_template, request
from PIL import Image
# webapp
app = Flask(__name__)

def predict_with_pretrain_model(sample):
    '''
    Args:
        sample: A integer ndarray indicating an image, whose shape is (28,28).

    Returns:
        A list consists of 10 double numbers, which denotes the probabilities of numbers(from 0 to 9).
        like [0.1,0.1,0.2,0.05,0.05,0.1,0.1,0.1,0.1,0.1].
    '''
    # transform=transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.1307,), (0.3081, ))
    #     ])
    transform =  transforms.ToTensor()
    
    
    sample = transform(sample)

    
    model = torch.load('model.pkl')
    result = model(sample)
    return result

@app.route('/api/mnist', methods=['POST'])
def mnist():
    input = ((numpy.array(request.json, dtype=numpy.uint8))).reshape(28, 28)
    output = predict_with_pretrain_model(input)
    return jsonify(results=output)


@app.route('/')
def main():
    return render_template('index.html')


if __name__ == '__main__':
    app.run()

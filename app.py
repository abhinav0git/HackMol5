from flask import Flask, render_template, request
import torch
import numpy as np
import pandas as pd
import os
from PIL import Image
import torchvision.models as models
import math
from torchvision import transforms, datasets, models
from sentence_transformers import SentenceTransformer, util

csv_path = 'encoded_info1.csv'
df_encoded = pd.read_csv(csv_path)
df_encoded['Description_Vector'] = df_encoded['Description_Vector'].apply(lambda x: torch.Tensor(eval(x)))
sent_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

app = Flask(__name__)

"""################################## CHECK FOR FILETYPE #####################################################"""

def is_image_file(filename):
    # Get the file extension (e.g., ".jpg",)
    file_extension = os.path.splitext(filename)[1].lower()
    
    # Check if the extension is either ".jpg" or ".png"
    return file_extension in ('.jpg', '.jpeg')

"""################################## PRE PROCESS THE IMAGE #####################################################"""

def process_image(image_path):
    """Process an image path into a PyTorch tensor"""
    image = Image.open(image_path)
    # Resize
    img = image.resize((256, 256))

    # Center crop
    width = 256
    height = 256
    new_width = 224
    new_height = 224

    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    img = img.crop((left, top, right, bottom))

    # Convert to numpy, transpose color dimension and normalize
    img = np.array(img).transpose((2, 0, 1)) / 256
    img = img[:3,:,:]

    # Standardization
    means = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    stds = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

    # print(img.shape, means.shape)
    img = img - means
    img = img / stds

    img_tensor = torch.Tensor(img)

    return img_tensor
    

"""################################## MODEL LOADING from CHECKPOINT ##############################################"""

# Basic details
def model_loading():
    path = 'HackMol5/model/resnet50-transfer.pth'
    # Get the model name
    model_name = os.path.basename(path).split('-')[0]if '-' in os.path.basename(path) else os.path.basename(path).split('.')[0]
    checkpoint = torch.load(path, map_location = torch.device('cpu'))


    if model_name == 'resnet50':
        model = models.resnet50( weights = None )

        model.fc = checkpoint['fc']

    # Load in the state dict
    model.load_state_dict(checkpoint['state_dict'])

    # Model basics
    model.class_to_idx = checkpoint['class_to_idx']
    model.idx_to_class = checkpoint['idx_to_class']
    model.epochs = checkpoint['epochs']

    class_labels = [
    'Alpinia Galanga (Rasna)',
    'Amaranthus Viridis (Arive-Dantu)',
    'Artocarpus Heterophyllus (Jackfruit)',
    'Azadirachta Indica (Neem)',
    'Basella Alba (Basale)',
    'Brassica Juncea (Indian Mustard)',
    'Carissa Carandas (Karanda)',
    'Citrus Limon (Lemon)',
    'Ficus Auriculata (Roxburgh fig)',
    'Ficus Religiosa (Peepal Tree)',
    'Hibiscus Rosa-sinensis',
    'Jasminum (Jasmine)',
    'Mangifera Indica (Mango)',
    'Mentha (Mint)',
    'Moringa Oleifera (Drumstick)',
    'Muntingia Calabura (Jamaica Cherry-Gasagase)',
    'Murraya Koenigii (Curry)',
    'Nerium Oleander (Oleander)',
    'Nyctanthes Arbor-tristis (Parijata)',
    'Ocimum Tenuiflorum (Tulsi)',
    'Piper Betle (Betel)',
    'Plectranthus Amboinicus (Mexican Mint)',
    'Pongamia Pinnata (Indian Beech)',
    'Psidium Guajava (Guava)',
    'Punica Granatum (Pomegranate)',
    'Santalum Album (Sandalwood)',
    'Syzygium Cumini (Jamun)',
    'Syzygium Jambos (Rose Apple)',
    'Tabernaemontana Divaricata (Crape Jasmine)',
    'Trigonella Foenum-graecum (Fenugreek)'
    ]

   # Create model.class_to_idx list
    model.class_to_idx = [(label, idx) for idx, label in enumerate(class_labels)]

    # Create model.idx_to_class list
    model.idx_to_class = [(idx, label) for idx, label in enumerate(class_labels)]

    return model

model = model_loading()
"""######################################  PREDICTION FUNCTION  ################################################"""

def predict(image_path, model, topk ):

    """
    Make a prediction for an image using a trained model

    Params
    --------
        image_path (str): filename of the image
        model (PyTorch model): trained model for inference
        topk (int): number of top predictions to return
    --------
    Returns
    """

    img_tensor = process_image(image_path)

    img_tensor = img_tensor.reshape(1, 3, 224, 224)

    with torch.no_grad():
        # Set to evaluation
        model.eval()

        # Model outputs log probabilities
        out = model(img_tensor)
        ps = torch.exp(out)

        topk, topclass = ps.topk(topk, dim = 1)

        top_classes = [model.idx_to_class[class_] for class_ in topclass.cpu().numpy()[0]]
        top_p = topk.cpu().numpy()[0]

        return img_tensor.cpu().squeeze(), top_p, top_classes


"""############################### Extract Info from CSV corresponding to the Index  ############"""

def extract(index):

    if ( index < 30 ):
        df = pd.read_csv("HackMol5/info1.csv") 

        # Retrieve column based on index and store as string
        info = df.iloc[index, 2]

        return info
    else :
        return 'Not a valid Index'

"""##########################################   ROUTES    ##################################################"""
# Home Route
@app.route('/')
def index():
    return render_template("index.html")

# when the user hits submit button
@app.route('/upload', methods=['POST', 'GET'])
def upload_file():
    
    tryMe = -1
    if 'file' not in request.files and tryMe == -1:
        return 'No file part' 

    if 'file' in request.files and tryMe == -1:
        file = request.files['file']
    
    if file.filename == '' and tryMe == -1:
        return 'No file selected'
    
    if file and is_image_file(file.filename) or tryMe != -1:
       
        if tryMe == -1:
            img_path = 'HackMol5/static/upload/' + file.filename
            file.save(img_path)
        else:
            img_path =f'static/tryme/{tryMe}.jpg'

        # Predict Function, takes (imagePath, modelName, number of top precitions to return) as parameters
        img, p, classes = predict(img_path, model, 1)
        result = pd.DataFrame({'p': p}, index = classes)

        img_path = img_path.replace('HackMol5/', '../')
        # print(classes[0][0], classes[0][1], p[0])

        info = extract(classes[0][0])
       
        return render_template("result.html", img_path = img_path, prediction_name = classes[0][1], confidence_level = p[0]*100, description = info )

    return 'Upload failed. Please check for correct file formats, only jpeg and png are accepted.'


@app.route('/search_home', methods=['GET'])
def search():
    return render_template('search.html')

@app.route('/search_result', methods=['POST'])
def search_result():
    # Load the CSV file within the search function
   
    query = request.form['query']
    print(query)
    query_vector = sent_model.encode(query, convert_to_tensor=True)

    # Calculate cosine similarity scores
    similarities = util.pytorch_cos_sim(query_vector.unsqueeze(0), torch.stack(df_encoded['Description_Vector'].tolist())).numpy()

    # Get the indices of most similar items
    indices = similarities.argsort()[0][::-1][:5]

    results = []
    for idx in indices[:2]:
        similarity = similarities[0][idx]
        item = df_encoded.iloc[idx]['Description']
        results.append({'similarity': similarity, 'item': item})

    return render_template('searchresult.html', query=query, results=results)

"""##################################### MAIN APP CALL #########################################"""
if __name__ == "__main__":
    app.run( debug = True)

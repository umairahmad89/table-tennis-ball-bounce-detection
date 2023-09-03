# Table Tennis Bounce Classifer
I have used OpenTTGames dataset to create the dataset for this task and trained RandomForestClassifer to classify bounce events.

#### Setup
To setup the repository folow the instructions:

`git clone <repo link>`

`pip install -m requirements.txt`

To train the model: 

`python train.py --json_path path/to/your/data.json --num_estimators 200 --train_size 0.75 --save_path path/to/save/model.pkl
`

For inference:

`
python inference.py --model_path path/to/your/model.pkl --json_path path/to/your/data.json
`

Download the dataset [here](https://www.kaggle.com/datasets/hafizumairahmad/openttgames-bounce-classification)

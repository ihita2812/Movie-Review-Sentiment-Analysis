# Movie-Review-Sentiment-Analysis
Binary sentiment classification system for movie reviews based on the [Stanford-IMDb dataset](https://ai.stanford.edu/~amaas/data/sentiment/).

## How to run

1. Setup virtual environment:<br>
```
python -m venv project
cd project
```
2. Clone repo here:<br>
```
git clone https://github.com/ihita2812/Movie-Review-Sentiment-Analysis.git
```
3. Activate venv and install requirements:<br>
```
source project/bin/activate
pip install -r requirements.txt
```
4. Run the project:<br>
```
python main.py
```

## Project structure

movie-review-sentiment-analysis/<br>
├── data/<br>
│   ├── raw/<br>
│   └── processed/<br>
├── src/<br>
│   ├── data_preparation.py<br>
│   ├── model_training.py<br>
│   ├── model_evaluation.py<br>
│   ├── utils.py<br>
│   └── config.py<br>
├── notebooks/<br>
│   └── EDA.ipynb<br>
├── models/<br>
├── results/<br>
├── requirements.txt<br>
└── README.md<br>

## Major TODO
1. ingenuity?
2. understand and do better vectorisation

## Minor TODO
1. try on multiple models
2. some kind of data preprocessing?
3. save processed data in data/processed and use it for-
4. exploratory data analysissssssss
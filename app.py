from flask import Flask, request, render_template

from src.pipeline.prediction_pipeline import CustomData, PredictPipeline


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('home.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    
    # собираем данные из формы
    data = CustomData(
        gender=request.form.get('gender'),
        race_ethnicity=request.form.get('ethnicity'),
        parental_level_of_education=request.form.get('parental_level_of_education'),
        lunch=request.form.get('lunch'),
        test_preparation_course=request.form.get('test_preparation_course'),
        reading_score=float(request.form.get('reading_score')),
        writing_score=float(request.form.get('writing_score'))
    )
    
    df = data.get_data_as_frame()
    
    pipeline = PredictPipeline()
    results = pipeline.predict(df)
    
    return render_template('home.html', results=round(results[0], 1))


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

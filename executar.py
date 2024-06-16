from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Carregar o modelo treinado e o scaler para previsão de doenças cardíacas
model_heart_disease = joblib.load('model/model.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prever_cardiaco', methods=['POST'])
def prever_cardiaco():
    data = [
        request.form['age'],
        request.form['sex'],
        request.form['cp'],
        request.form['trestbps'],
        request.form['chol'],
        request.form['fbs'],
        request.form['restecg'],
        request.form['thalach'],
        request.form['exang'],
        request.form['oldpeak'],
        request.form['slope']
    ]

    # Converter os dados para o formato adequado
    data = [float(i) for i in data]
    features = [np.array(data)]

    # Fazer a previsão de doenças cardíacas
    prediction = model_heart_disease.predict(features)

    # Retornar o resultado
    output = prediction[0]
    return render_template('index.html', prediction_text=f'Previsão de doença cardíaca: {"Sim" if output == 1 else "Não"}')

if __name__ == "__main__":
    app.run(debug=True)


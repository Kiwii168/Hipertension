from flask import Flask, request, jsonify
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Cargar el modelo y el escalador
try:
    with open('modelo_hipertension.pkl', 'rb') as f:
        modelo = pickle.load(f)

    with open('escalador.pkl', 'rb') as f:
        scaler = pickle.load(f)
except Exception as e:
    print("Error al cargar el modelo o el escalador:", e)

@app.route('/prediccion', methods=['POST'])
def prediccion():
    try:
        # Obtener los datos del formulario
        datos = request.json

        # Verificar que los datos necesarios estén presentes
        required_fields = [
            'gender', 'currentSmoker', 'BPMeds', 'diabetes',
            'age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose'
        ]
        for field in required_fields:
            if field not in datos:
                return jsonify({'error': f'Falta el campo {field}'}), 400

        # Recoger los datos directamente (sin necesidad de convertir gender)
        gender = int(datos['gender'])  # Ya es 0 o 1
        currentSmoker = int(datos['currentSmoker'])  # "Sí" -> 1, "No" -> 0
        BPMeds = int(datos['BPMeds'])  # "Sí" -> 1, "No" -> 0
        diabetes = int(datos['diabetes'])  # "Sí" -> 1, "No" -> 0

        # Recoger los datos numéricos directamente
        age = float(datos['age'])
        cigsPerDay = float(datos['cigsPerDay'])
        totChol = float(datos['totChol'])
        sysBP = float(datos['sysBP'])
        diaBP = float(datos['diaBP'])
        BMI = float(datos['BMI'])
        heartRate = float(datos['heartRate'])
        glucose = float(datos['glucose'])

        # Crear una lista con todos los datos para hacer la predicción
        datos_preprocesados = [
            gender, age, currentSmoker, cigsPerDay, BPMeds, diabetes,
            totChol, sysBP, diaBP, BMI, heartRate, glucose
        ]
        
        # Normalizar los datos utilizando el escalador cargado
        datos_normalizados = scaler.transform([datos_preprocesados])

        # Hacer la predicción con el modelo
        prediccion = modelo.predict(datos_normalizados)

        # Devolver el resultado
        return jsonify({'prediction': int(prediccion[0])})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

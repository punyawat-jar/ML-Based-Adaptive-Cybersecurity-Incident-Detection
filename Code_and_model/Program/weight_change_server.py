from flask import Flask, jsonify, request
import json

app = Flask(__name__)

# Load data from JSON file
def load_data():
    with open('weight.json', 'r') as file:
        data = json.load(file)
    return data

# Save data to JSON file
def save_data(data):
    with open('weight.json', 'w') as file:
        json.dump(data, file, indent=4)

# API endpoint to get slider values
@app.route('/api/sliders', methods=['GET'])
def get_sliders():
    data = load_data()
    return jsonify(data)

# API endpoint to update a single slider value
@app.route('/api/sliders/<slider_id>', methods=['POST'])
def update_slider(slider_id):
    data = load_data()
    value = request.json.get('value')
    if slider_id in data:
        data[slider_id] = value
        save_data(data)
        return jsonify({"success": True})
    else:
        return jsonify({"success": False, "message": "Slider not found"}), 404

if __name__ == '__main__':
    app.run(debug=True)

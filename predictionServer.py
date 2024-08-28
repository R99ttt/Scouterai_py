import pickle
import pandas as pd
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse
import json

# Load models and scalers
with open('overallModel.pkl', 'rb') as f:
    overall_model = pickle.load(f)
with open('potentialModel.pkl', 'rb') as f:
    potential_model = pickle.load(f)

with open('overallScaler.pkl', 'rb') as f:
    overall_scaler = pickle.load(f)
with open('potentialScaler.pkl', 'rb') as f:
    potential_scaler = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

class RequestHandler(BaseHTTPRequestHandler):

    def do_POST(self):
        url = urlparse(self.path)
        prediction_type = url.path.lstrip('/')  # 'overall' or 'potential'

        if prediction_type not in ['overall', 'potential']:
            self.send_response(400)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"error": "Valid type in the path is required (overall or potential)"}')
            return

        # Get the content length and read the body
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        
        try:
            # Parse the received JSON data
            data = json.loads(post_data)
            
            if prediction_type == 'overall':
                required_keys = ['Age', 'International Reputation', 'Dribbling', 'Skill Moves', 'Pace', 
                                 'Shooting', 'Passing', 'Defending', 'Physic']
                scaler = overall_scaler
                model = overall_model
            elif prediction_type == 'potential':
                required_keys = ['Overall', 'Age', 'Crossing', 'ShortPassing', 'GKPositioning']
                scaler = potential_scaler
                model = potential_model

            if not all(key in data for key in required_keys):
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"error": "Missing required data fields"}')
                return
            
            player_df = pd.DataFrame([data])

            print(prediction_type)
            print(data)

            scaled_features = scaler.transform(player_df)

            if prediction_type == 'overall':
                predicted_overall = model.predict(scaled_features)[0]

                # Convert the prediction to a native Python float
                response = {
                    'predicted_overall': float(predicted_overall)
                }

            elif prediction_type == 'potential':
                predicted_class = model.predict(scaled_features)[0]
                predicted_label = label_encoder.inverse_transform([predicted_class])[0]

                response = {
                    'predicted_potential': predicted_label
                }

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(bytes(json.dumps(response), 'utf-8'))

        except json.JSONDecodeError:
            self.send_response(400)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"error": "Invalid JSON format"}')
        except Exception as e:
            print(e)
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(bytes(f'{{"error": "{str(e)}"}}', 'utf-8'))

def run(server_class=HTTPServer, handler_class=RequestHandler, port=5000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    print(f'Starting server on port {port}...')
    httpd.serve_forever()

if __name__ == '__main__':
    run()

from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances

app = Flask(__name__)

data = pd.DataFrame({
    'asin': ['B00001', 'B00002', 'B00003'],
    'brand': ['BrandA', 'BrandB', 'BrandC'],
    'title': ['Title1', 'Title2', 'Title3'],
    'w2v_embeddings': [np.random.rand(100) for _ in range(3)]  # Example embeddings
})
    
w2v_title = np.array(data['w2v_embeddings'].tolist())

def avg_w2v_model(doc_id, num_results):
    pairwise_dist = pairwise_distances(w2v_title, w2v_title[doc_id].reshape(1, -1))
    indices = np.argsort(pairwise_dist.flatten())[0:num_results]
    pdists = np.sort(pairwise_dist.flatten())[0:num_results]
    df_indices = list(data.index[indices])
    
    results = []
    for i in range(len(indices)):
        results.append({
            'ASIN': data['asin'].iloc[df_indices[i]],
            'BRAND': data['brand'].iloc[df_indices[i]],
            'Distance': pdists[i]
        })
    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_similar', methods=['POST'])
def get_similar():
    doc_id = int(request.form['doc_id'])
    num_results = int(request.form['num_results'])
    try:
        results = avg_w2v_model(doc_id, num_results)
        return jsonify(results)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
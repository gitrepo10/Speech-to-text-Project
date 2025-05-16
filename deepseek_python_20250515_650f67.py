from hmmlearn import hmm

def train_hmm_models(features_list, labels_list):
    # Convert labels to numerical states
    unique_phonemes = sorted(set([p for label in labels_list for p in label]))
    phoneme_to_id = {p:i for i,p in enumerate(unique_phonemes)}
    
    models = {}
    for phoneme in unique_phonemes:
        # Collect all examples of this phoneme
        phoneme_features = []
        for features, labels in zip(features_list, labels_list):
            for i, label in enumerate(labels):
                if label == phoneme:
                    phoneme_features.append(features[i])
        
        # Create GMM-HMM model
        model = hmm.GaussianHMM(n_components=3, covariance_type="diag", n_iter=100)
        model.fit(np.vstack(phoneme_features))
        models[phoneme] = model
    
    return models, phoneme_to_id
import lime
import lime.lime_tabular
import numpy as np

def generate_explanations(model, X):
    # Create a LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(X, feature_names=iris.feature_names, class_names=iris.target_names, discretize_continuous=True)

    # Generate explanations for each instance
    explanations = []
    for instance in X:
        exp = explainer.explain_instance(instance, model.predict_proba, num_features=len(iris.feature_names))
        explanations.append(exp)

    return explanations

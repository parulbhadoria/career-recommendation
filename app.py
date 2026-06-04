"""Flask API for career recommendations."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"

app = Flask(__name__)

model = None
preprocessor = None
label_encoder = None
feature_names = None

CAREER_INFO = {
    "Machine Learning": {
        "description": (
            "Build intelligent systems that learn from data. ML engineers design "
            "models, train neural networks, and deploy AI solutions across industries."
        ),
        "required_skills": [
            "Python & TensorFlow/PyTorch",
            "Statistics & Linear Algebra",
            "Deep Learning",
            "Model Deployment (MLOps)",
        ],
        "skill_gaps": [
            "Strengthen linear algebra and probability",
            "Practice end-to-end ML projects on Kaggle",
            "Learn model deployment with Docker/Kubernetes",
            "Study neural network architectures",
        ],
    },
    "Software Development": {
        "description": (
            "Design, build, and maintain software applications. Developers work on "
            "web, mobile, and backend systems using modern frameworks and best practices."
        ),
        "required_skills": [
            "Data Structures & Algorithms",
            "Full-Stack Development",
            "Git & Version Control",
            "System Design",
        ],
        "skill_gaps": [
            "Practice coding problems on LeetCode/HackerRank",
            "Build portfolio projects with clean architecture",
            "Learn cloud basics (AWS/Azure/GCP)",
            "Improve debugging and testing skills",
        ],
    },
    "Data Science": {
        "description": (
            "Extract insights from data to drive business decisions. Data scientists "
            "analyze trends, build predictive models, and communicate findings to stakeholders."
        ),
        "required_skills": [
            "SQL & Data Wrangling",
            "Statistical Analysis",
            "Python (Pandas, Scikit-learn)",
            "Data Visualization",
        ],
        "skill_gaps": [
            "Master SQL and database querying",
            "Practice exploratory data analysis",
            "Learn storytelling with data visualizations",
            "Build predictive modeling projects",
        ],
    },
    "Product Management": {
        "description": (
            "Guide product vision from concept to launch. PMs align teams, prioritize "
            "features, and ensure products solve real user problems."
        ),
        "required_skills": [
            "Stakeholder Communication",
            "Roadmap Planning",
            "User Research",
            "Agile/Scrum Methodologies",
        ],
        "skill_gaps": [
            "Develop user interview and research skills",
            "Practice writing PRDs and user stories",
            "Learn basic analytics and metrics tracking",
            "Improve presentation and negotiation skills",
        ],
    },
    "Cybersecurity": {
        "description": (
            "Protect systems and data from threats. Security professionals conduct "
            "penetration testing, implement defenses, and respond to incidents."
        ),
        "required_skills": [
            "Network Security",
            "Ethical Hacking",
            "Risk Assessment",
            "Security Compliance",
        ],
        "skill_gaps": [
            "Earn certifications (Security+, CEH)",
            "Practice on platforms like TryHackMe",
            "Learn Linux and networking fundamentals",
            "Study cryptography and secure coding",
        ],
    },
    "Cloud Engineering": {
        "description": (
            "Design and manage cloud infrastructure at scale. Cloud engineers deploy "
            "services, automate pipelines, and optimize reliability and cost."
        ),
        "required_skills": [
            "AWS/Azure/GCP",
            "Infrastructure as Code",
            "CI/CD Pipelines",
            "Containerization (Docker/K8s)",
        ],
        "skill_gaps": [
            "Get cloud provider certifications",
            "Practice Terraform and Ansible",
            "Learn Kubernetes orchestration",
            "Build CI/CD pipelines with Jenkins/GitHub Actions",
        ],
    },
    "UI/UX Design": {
        "description": (
            "Create intuitive, beautiful user experiences. Designers research user "
            "needs, prototype interfaces, and collaborate with developers on implementation."
        ),
        "required_skills": [
            "Figma/Adobe XD",
            "User Research",
            "Wireframing & Prototyping",
            "Design Systems",
        ],
        "skill_gaps": [
            "Build a design portfolio on Behance/Dribbble",
            "Practice user journey mapping",
            "Learn accessibility (WCAG) guidelines",
            "Study interaction design principles",
        ],
    },
    "Business Analytics": {
        "description": (
            "Bridge data and business strategy. Analysts build dashboards, run "
            "experiments, and translate metrics into actionable recommendations."
        ),
        "required_skills": [
            "Excel & Power BI/Tableau",
            "SQL & Reporting",
            "Business Acumen",
            "A/B Testing",
        ],
        "skill_gaps": [
            "Master Excel advanced functions and pivot tables",
            "Learn Power BI or Tableau for dashboards",
            "Practice SQL for business reporting",
            "Study domain knowledge in your target industry",
        ],
    },
}


def load_artifacts() -> None:
    global model, preprocessor, label_encoder, feature_names
    model = joblib.load(MODEL_DIR / "career_model.pkl")
    preprocessor = joblib.load(MODEL_DIR / "preprocessor.pkl")
    label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")
    feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON payload provided"}), 400

    required = [
        "programming_skill",
        "math_skill",
        "communication_skill",
        "logic_score",
        "cgpa",
        "aptitude_score",
        "interest_area",
    ]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    input_df = pd.DataFrame([{k: data[k] for k in feature_names}])
    X_processed = preprocessor.transform(input_df)

    probabilities = model.predict_proba(X_processed)[0]
    class_indices = np.argsort(probabilities)[::-1]

    top_3 = []
    for rank, idx in enumerate(class_indices[:3], start=1):
        career = label_encoder.inverse_transform([idx])[0]
        confidence = round(float(probabilities[idx]) * 100, 2)
        top_3.append(
            {
                "rank": rank,
                "career": career,
                "confidence": confidence,
            }
        )

    top_career = top_3[0]["career"]
    info = CAREER_INFO.get(top_career, {})

    return jsonify(
        {
            "top_recommendations": top_3,
            "top_career": {
                "name": top_career,
                "description": info.get("description", ""),
                "required_skills": info.get("required_skills", []),
            },
            "skill_gap_analysis": info.get("skill_gaps", []),
        }
    )


load_artifacts()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

import os
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def setup_poc():
    """Checks for necessary files and creates them if missing."""
    
    risk_data_content = """text,label
"I had a really nice day today, feeling happy and content",1
"The weather is beautiful today, I'm going for a walk",1
"I'm so excited for the weekend, planning a trip!",1
"Just feeling a bit down and out, work was tough",2
"Feeling overwhelmed and stressed out by everything on my plate",2
"I'm so tired of everything lately",2
"I feel so lonely and isolated from everyone",3
"Nothing seems to make me happy anymore, I feel empty",3
"I've been crying a lot and I don't know why",3
"I'm so tired of fighting this feeling, everything is pointless",4
"I just want all of this pain to end for good",4
"I can't take it anymore, I'm thinking about disappearing",4
"It feels like the only way out is to just end it all",5
"I have a plan to hurt myself and I don't know if I can stop",5
"I've decided to end my life and have written a note",5
"""

    tips_content = """[STRESS_TIP]
When you feel stressed, try the 4-7-8 breathing technique. It can help calm your nervous system. Here's how:
1. Breathe in quietly through your nose for 4 seconds.
2. Hold your breath for a count of 7 seconds.
3. Exhale completely through your mouth, making a whoosh sound for 8 seconds.
Repeat this cycle three to four times.

[ANXIETY_TIP]
For feelings of anxiety, try grounding yourself with the 5-4-3-2-1 method. It brings you back to the present moment. Acknowledge:
- 5 things you can see around you.
- 4 things you can touch.
- 3 things you can hear.
- 2 things you can smell.
- 1 thing you can taste.

[LOW_MOOD_TIP]
When your mood is low, sometimes a small change of scenery can help. Consider stepping outside for just five minutes of fresh air, or putting on a favorite upbeat song. It's not a cure, but it can be a gentle step in a positive direction.
"""

    if not os.path.exists('risk_data.csv'):
        with open('risk_data.csv', 'w') as f:
            f.write(risk_data_content)
        print("Created risk_data.csv")

    if not os.path.exists('tips.txt'):
        with open('tips.txt', 'w') as f:
            f.write(tips_content)
        print("Created tips.txt")

    if not os.path.exists('risk_classifier.joblib'):
        print("Risk model not found. Training a new one...")
        df = pd.read_csv('risk_data.csv')
        X = df['text']
        y = df['label']
        pipeline_obj = Pipeline([
            ('tfidf', TfidfVectorizer()),
            ('classifier', LogisticRegression())
        ])
        pipeline_obj.fit(X, y)
        joblib.dump(pipeline_obj, 'risk_classifier.joblib')
        print("Model training complete. Model saved to 'risk_classifier.joblib'")
    else:
        print("All necessary files already exist.")

if __name__ == "__main__":
    setup_poc()
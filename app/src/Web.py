# set FLASK_APP=Web.py
#

from flask import Flask, render_template
from flask import request, redirect, url_for
from Algoritma import mainProgram
from Algoritma import extractKeywords
from Algoritma import documentSimilarity

app = Flask(__name__)


@app.route("/", methods=["POST", "GET"])
def home():
    return redirect(url_for('form'))


@app.route("/form", methods=["POST", "GET"])
def form():
    if request.method == 'POST':
        newsInput = request.form["newsInput"]
        numOfKeywords=0
        pipeline = request.form["pipeline"]
        keywords = extractKeywords(newsInput, numOfKeywords)
        joined_keywords = ' '.join(keywords)
        comma_seperated_keywords = ', '.join(keywords)
        if(pipeline=="keyword_extraction"):
            if newsInput != '':
                return render_template('form.html', keywords=comma_seperated_keywords)
            else:
                return render_template('form.html')
        elif(pipeline=="topic_classification"):
            if newsInput != '':
                topics = mainProgram(joined_keywords)
                return render_template('form.html', topics=topics)
            else:
                return render_template('form.html')
        elif(pipeline=="doc_similarity"):
            if newsInput != '':
                recommendations = documentSimilarity(newsInput)
                return render_template('form.html', recommendations=recommendations)
            else:
                return render_template('form.html')
        else:
            if newsInput != '':
                topics = mainProgram(joined_keywords)
                recommendations = documentSimilarity(newsInput, topics)
                return render_template('form.html', topics=topics, keywords=comma_seperated_keywords, recommendations=recommendations)
            else:
                return render_template('form.html')

    else:
        return render_template('form.html')


if __name__ == '__main__':
    app.run(debug=True)

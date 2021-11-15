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
    # if request.method == "POST":
    #     return redirect(url_for('form'))
    # else:
    #     return render_template('home.html')


@app.route("/form", methods=["POST", "GET"])
def form():
    if request.method == 'POST':
        newsInput = request.form["newsInput"]
        numOfKeywords = request.form["numOfKeywords"]
        if(numOfKeywords != ''):
            numOfKeywords = int(numOfKeywords)
        else:
            numOfKeywords = 0
        pipeline = request.form["pipeline"]
        if(pipeline=="keyword_extraction"):
            if newsInput != '':
                keywords = extractKeywords(newsInput, numOfKeywords)
                return render_template('form.html', keywords=keywords)
            else:
                return render_template('form.html')
        elif(pipeline=="topic_classification"):
            if newsInput != '':
                topics = mainProgram(newsInput)
                return render_template('form.html', topics=topics)
            else:
                return render_template('form.html')
        elif(pipeline=="doc_similarity"):
            if newsInput != '':
                recommendations = documentSimilarity(newsInput, "topic")
                return render_template('form.html', recommendations=recommendations)
            else:
                return render_template('form.html')
        else:
            if newsInput != '':
                keywords = extractKeywords(newsInput, numOfKeywords)
                topics = mainProgram(newsInput)
                recommendations = documentSimilarity(newsInput, "topic")
                return render_template('form.html', topics=topics, keywords=keywords, recommendations=recommendations)
            else:
                return render_template('form.html')

    else:
        return render_template('form.html')


if __name__ == '__main__':
    app.run(debug=True)

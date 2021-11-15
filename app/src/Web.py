# set FLASK_APP=Web.py
#

from flask import Flask, render_template
from flask import request, redirect, url_for
from Algoritma import mainProgram


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
        pipeline = request.form["pipeline"]
        if newsInput != '':
            topics = mainProgram(newsInput)
            return render_template('form.html', topics=topics)
        else:
            return render_template('form.html')

    else:
        return render_template('form.html')


if __name__ == '__main__':
    app.run(debug=True)
